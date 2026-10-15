# [MODULE DOC]
# Purpose: Background checkpoint mover that shells out to the s5cmd CLI to upload a local checkpoint
# directory to S3, optionally delete it locally afterwards, or download a checkpoint from S3.
# Main contents: S3Mover class (a small state machine driven by polling update()).
# Fits in nanotron: created by the trainer when s3 upload is configured; typically one non-dummy
# mover per node, with all other processes using dummy=True.
# Import-time side effects: creates a module logger.
import glob
import json
import os
import subprocess
import time
from datetime import datetime
from enum import Enum
from typing import Optional, Union

import torch
from datasets.download.streaming_download_manager import xPath
from filelock import FileLock, Timeout

from nanotron import distributed as dist
from nanotron import logging
from nanotron.distributed import ProcessGroup
from nanotron.logging import human_format

logger = logging.get_logger(__name__)


class S3Mover:
    # [CLASS DOC]
    # Purpose: Non-blocking state machine IDLE -> UPLOADING -> (REMOVING_CHECKPOINT) -> IDLE, or
    # IDLE -> DOWNLOADING -> IDLE, each phase running one subprocess.
    # Key attributes: process (Popen, DummyPopen or None), state (S3MoverState), local_path and s3_path
    # (both normalized with trailing slash), full_local_path / full_s3_path (set by start_uploading),
    # s5cmd options, lock_file (FileLock held during upload/removal), stdout accumulated from the process,
    # start_time, cmd, num_uploaded_files, post_upload_callback and its outputs, dummy flag,
    # s3_path_direct_link (AWS console URL).
    # Invariant: only one background process at a time; new operations require state IDLE.
    # KNOWN ISSUE: the class docstring describes only uploading, although downloading is also supported (noted by the TODO).
    # TODO @eliebak update the doc to state that it also the function use to download it to the disk with start_downloading
    """Take care of uploading a checkpoint to S3 in the background and remove it from the disk.

    Args:
        local_path: Path to the checkpoints on the local disk
        s3_path: Path to the checkpoints on S3
        remove_after_upload: If True, remove the checkpoint from the disk after uploading it to S3
        s5cmd_numworkers: Number of workers to use for the s5cmd command
        s5cmd_concurrency: Concurrency to use for the s5cmd command
        s5cmd_path: Path to the s5cmd command
        dummy: If True, don't actually upload/remove/etc anything. Useful for simpler multi-processing node and only uploading from one process.

    Usage:
        # Create a mover - use dummy=True for all the process that shouldn't do anything (e.g. all but one per node)
        mover = S3Mover(local_path=/scratch/my-checkpoints,
                        s3_path=s3://my-bucket/my-checkpoints,
                        remove_after_upload=True,
                        s5cmd_numworkers=96,
                        s5cmd_concurrency=10,
                        s5cmd_path=/admin/user/my/bin/s5cmd,
                        dummy=False)

        while training:
            # from times to times update the state
            mover_status = mover.update()
            ...

            # When saving a checkpoint, check if the previous checkpoint has been uploaded and removed
            # in a distributed setting
    """

    class S3MoverState(Enum):
        # [CLASS DOC]
        # Purpose: Enum of mover states: IDLE, UPLOADING, DOWNLOADING, REMOVING_CHECKPOINT (string values equal to names).
        IDLE = "IDLE"
        UPLOADING = "UPLOADING"
        DOWNLOADING = "DOWNLOADING"
        REMOVING_CHECKPOINT = "REMOVING_CHECKPOINT"

    class DummyPopen:
        # [CLASS DOC]
        # Purpose: Stand-in for subprocess.Popen used in dummy mode; always reports successful completion with empty output.
        def __init__(self, *args, **kwargs):
            # [FUNCTION DOC]
            # Behavior: ignores all arguments. Inputs: any. Outputs: None. Exceptions: None. Side effects: None.
            pass

        def poll(self):
            # [FUNCTION DOC]
            # Behavior: returns 0 (finished successfully). Inputs: None. Outputs: int 0. Exceptions: None. Side effects: None.
            return 0

        def communicate(self):
            # [FUNCTION DOC]
            # Behavior: returns a pair of empty strings. Inputs: None. Outputs: tuple of two strings. Exceptions: None. Side effects: None.
            return ("", "")

    def __init__(
        self,
        local_path: xPath,
        s3_path: xPath,
        post_upload_callback: Optional[callable] = None,
        remove_after_upload: Optional[bool] = True,
        s5cmd_numworkers: Optional[int] = None,
        s5cmd_concurrency: Optional[int] = None,
        s5cmd_path: Optional[str] = None,
        s5cmd_credentials: Optional[str] = None,
        clean_up_local_on_start: bool = False,
        dummy: bool = False,
        s3_region: str = "us-east-1",
    ):
        # [FUNCTION DOC]
        # Behavior: stores options (s5cmd path defaults to "s5cmd"), warns if local_path is not under
        # /scratch/, normalizes both paths to end with a slash, splits s3_path into bucket and prefix to build
        # the console link, resets state, and if clean_up_local_on_start begins removing local checkpoint dirs.
        # Inputs: local_path, s3_path (xPath or str; s3_path like s3://bucket/prefix), post_upload_callback
        # (optional callable taking uploaded_files keyword), remove_after_upload (default True),
        # s5cmd_numworkers, s5cmd_concurrency, s5cmd_path, s5cmd_credentials (optional), clean_up_local_on_start
        # (default False), dummy (default False), s3_region (default "us-east-1").
        # Outputs: None.
        # Exceptions: ValueError if s3_path has no slash after the bucket (unpacking fails).
        # Side effects: logging; may spawn an rm subprocess.
        self.process: Optional[Union[subprocess.Popen, S3Mover.DummyPopen]] = None
        self.remove_after_upload = remove_after_upload
        self.s5cmd_numworkers = s5cmd_numworkers
        self.s5cmd_concurrency = s5cmd_concurrency
        self.s5cmd_path = s5cmd_path if s5cmd_path is not None else "s5cmd"
        self.s5cmd_credentials = s5cmd_credentials
        self.lock_file = None
        self.dummy = dummy
        self.s3_region = s3_region
        self.post_upload_callback = post_upload_callback
        self.post_upload_callback_outputs = None

        local_path = str(local_path)
        if not local_path.startswith("/scratch/"):
            self._warning(f"The local path is not on the scratch drive: {local_path}")
        if not local_path.endswith("/"):
            local_path += "/"

        s3_path = str(s3_path)
        if not s3_path.endswith("/"):
            s3_path += "/"

        self.local_path = local_path
        self.s3_path = s3_path

        s3_bucket, s3_prefix = s3_path.replace("s3://", "").split("/", maxsplit=1)
        self.s3_path_direct_link = f"https://s3.console.aws.amazon.com/s3/buckets/{s3_bucket}?region={self.s3_region}&prefix={s3_prefix}&showversions=false"

        self._reset_state()
        if clean_up_local_on_start:
            self._start_removing()

    def _warning(self, message):
        # [FUNCTION DOC]
        # HELPER: logging that is silenced for dummy movers.
        # Behavior: logs message at warning level unless dummy. Inputs: message (str). Outputs: None.
        # Exceptions: None. Side effects: logging.
        if self.dummy:
            return
        logger.warning(message)

    def _info(self, message):
        # [FUNCTION DOC]
        # HELPER: info-level counterpart of _warning. Behavior: logs at info level unless dummy.
        # Inputs: message. Outputs: None. Exceptions: None. Side effects: logging.
        if self.dummy:
            return
        logger.info(message)

    def _reset_state(self):
        # [FUNCTION DOC]
        # HELPER: returns the mover to IDLE.
        # Behavior: sets state IDLE, zeroes num_uploaded_files, releases and clears any lock, clears stdout,
        # start_time and cmd.
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised. Side effects: mutates self; releases file lock.
        self.state = self.S3MoverState.IDLE
        self.num_uploaded_files = 0
        if self.lock_file is not None:
            self._release_lock()
        self.lock_file = None
        self.stdout = ""
        self.start_time: datetime = None
        self.cmd = ""

    def _popen(self, cmd: list):
        # [FUNCTION DOC]
        # HELPER: launches a background command.
        # Behavior: clears stdout, records start time and cmd; in dummy mode returns a DummyPopen, else starts
        # a subprocess with stdout piped (stderr merged) and sets the pipe non-blocking.
        # Inputs: cmd (list of strings). Outputs: Popen or DummyPopen.
        # Exceptions: OSError if the executable cannot be launched. Side effects: spawns a process.
        self.stdout = ""
        self.start_time = datetime.now()
        self.cmd = cmd
        if self.dummy:
            return self.DummyPopen(cmd)
        else:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            os.set_blocking(process.stdout.fileno(), False)
            return process

    def _acquire_lock(self, file_path: str) -> bool:
        # [FUNCTION DOC]
        # HELPER: inter-process lock so only one mover handles a given checkpoint.
        # Behavior: in dummy mode returns True; otherwise derives a lock path by appending ".lock" (after
        # stripping a trailing slash), creates a FileLock and tries to acquire it within 1 second.
        # Inputs: file_path (str). Outputs: True if acquired, False on timeout (with a warning).
        # Exceptions: None explicitly raised. Side effects: creates lock file; sets self.lock_file.
        if self.dummy:
            return True
        if file_path.endswith("/"):
            lock_file_path = file_path[:-1] + ".lock"
        else:
            lock_file_path = file_path + ".lock"
        self.lock_file = FileLock(lock_file_path)
        try:
            self.lock_file.acquire(timeout=1)
        except Timeout:
            message = f"[S3] The checkpoint files {lock_file_path} are currently locked by another process. "
            self._warning(message)
            return False
        return True

    def get_state_as_int(self) -> int:
        # [FUNCTION DOC]
        # Behavior: maps IDLE, UPLOADING, DOWNLOADING, REMOVING_CHECKPOINT to 0, 1, 2, 3 respectively, otherwise -1.
        # Inputs: None. Outputs: int. Exceptions: None. Side effects: None.
        """Return the state as an int"""
        if self.state == self.S3MoverState.IDLE:
            return 0
        elif self.state == self.S3MoverState.UPLOADING:
            return 1
        elif self.state == self.S3MoverState.DOWNLOADING:
            return 2
        elif self.state == self.S3MoverState.REMOVING_CHECKPOINT:
            return 3
        else:
            return -1

    def _release_lock(self):
        # [FUNCTION DOC]
        # HELPER: releases the held FileLock if any (no-op in dummy mode).
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised. Side effects: releases lock.
        if self.dummy:
            return
        if self.lock_file is not None and self.lock_file.is_locked:
            self.lock_file.release()

    def get_current_stdout(self) -> str:
        # [FUNCTION DOC]
        # Behavior: returns empty string if there is no real process; otherwise non-blockingly reads available
        # bytes (treating a closed pipe as empty), appends their decoded text to self.stdout and returns the
        # accumulated stdout.
        # Inputs: None. Outputs: str. Exceptions: None explicitly raised. Side effects: mutates self.stdout.
        """Return the current stdout of the process if any"""
        if self.process is None or isinstance(self.process, self.DummyPopen):
            return ""
        try:
            stdout = self.process.stdout.read()
        except ValueError:
            stdout = ""  # The buffer is already closed: "ValueError: read of closed file"
        if stdout:
            self.stdout += stdout.decode()
        return self.stdout

    def wait_for_completion(self):
        # [FUNCTION DOC]
        # Behavior: polls update every 0.5 seconds until the state is IDLE (blocking, local only).
        # Inputs: None. Outputs: None. Exceptions: None explicitly raised.
        # Side effects: sleeps; drives state transitions and subprocesses.
        # KNOWN ISSUE: if a subprocess fails (non-zero return code) update never resets the state, so this loops forever.
        while self.state != self.S3MoverState.IDLE:
            _ = self.update()
            time.sleep(0.5)

    def distributed_wait_for_completion(self, group: Optional[ProcessGroup] = None):
        # [FUNCTION DOC]
        # Behavior: on the given group (default world group), each rank computes whether its previous save is
        # finished, all-gathers these flags as CUDA tensors and barriers; warns if not everyone is done; then
        # loops (sleeping 1 second per iteration, logging progress and the last stdout line) repeating the
        # all-gather until all ranks report finished.
        # Inputs: group (optional ProcessGroup).
        # Outputs: None. Exceptions: None explicitly raised.
        # Side effects: collectives (all_gather and barrier on the default group) every iteration; logging; sleeps.
        # KNOWN ISSUE: barrier is called on the default group rather than group, which deadlocks if group is a strict subset of ranks.
        """Wait for the previous checkpoint to be fully uploaded and removed in a distributed setting.
        Will wait for all process to be ready
        """
        if group is None:
            group = dist.torch_dist.distributed_c10d._get_default_group()

        test_tensor = torch.tensor([self.is_previous_save_finished()], device=torch.device("cuda"))
        test_tensor_list = [torch.zeros_like(test_tensor) for _ in range(group.size())]
        dist.all_gather(test_tensor_list, test_tensor, group=group, async_op=False)
        dist.barrier()
        all_saved = sum(bool(tensor.item()) for tensor in test_tensor_list)
        if all_saved != group.size() and self.state != self.S3MoverState.IDLE:
            self._warning(
                f"Waiting previous checkpoint saving is finished - S3Mover {dist.get_rank(group)} still in {self.state} state.",
            )
        while all_saved != group.size():
            stdout = self.get_current_stdout()
            stdout_lines = [lst for lst in stdout.split("\n") if lst]
            if self.state != self.S3MoverState.IDLE:
                self._warning(
                    f"[S3] Waiting {self.state.value}: {all_saved} / {group.size()}. Stdout: {len(stdout_lines)} end: {stdout_lines[-1:]}",
                )
            # sync all our saves on NCCL we could do a dist barrier later but this helps us not losing NCCL connections down the line
            test_tensor = torch.tensor([self.is_previous_save_finished()], device=torch.device("cuda"))
            test_tensor_list = [torch.zeros_like(test_tensor) for _ in range(group.size())]
            dist.all_gather(test_tensor_list, test_tensor, group=group, async_op=False)
            dist.barrier()
            all_saved = sum(bool(tensor.item()) for tensor in test_tensor_list)
            time.sleep(1)  # TODO @nouamane: make this configurable

    def is_previous_save_finished(self) -> bool:
        # [FUNCTION DOC]
        # Behavior: calls update and returns whether the state is IDLE.
        # Inputs: None. Outputs: bool. Exceptions: None explicitly raised. Side effects: same as update.
        """Return True if a potential previous checkpoint has been fully uploaded to S3
        and removed from the drive
        """
        self.update()
        return self.state == self.S3MoverState.IDLE

    def _start_downloading(self, sub_folder: Optional[str] = None) -> (bool, str):
        # [FUNCTION DOC]
        # HELPER: begins the DOWNLOADING phase.
        # Behavior: logs, builds an s5cmd command with json output plus optional credentials file, numworkers,
        # "cp", optional concurrency, source s3_path followed by a wildcard and destination local_path;
        # launches it and sets state DOWNLOADING.
        # Inputs: sub_folder (unused). Outputs: True.
        # Exceptions: None explicitly raised. Side effects: spawns process; logging.
        # KNOWN ISSUE: annotated to return a pair but returns a single bool; sub_folder is ignored.
        self._warning(
            f"[S3] Downloading checkpoint in background from {self.s3_path} to {self.local_path} (direct link: {self.s3_path_direct_link})"
        )
        cmd = [self.s5cmd_path, "--json"]
        if self.s5cmd_credentials is not None:
            cmd += ["--credentials-file", self.s5cmd_credentials]
        if self.s5cmd_numworkers is not None:
            cmd += ["--numworkers", str(self.s5cmd_numworkers)]
        cmd += ["cp"]
        if self.s5cmd_concurrency is not None:
            cmd += ["--concurrency", str(self.s5cmd_concurrency)]
        cmd += [self.s3_path + "*", self.local_path]

        self.process = self._popen(cmd)
        self.state = self.S3MoverState.DOWNLOADING

        return True

    def _post_downloading(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: summarizes a finished download.
        # Behavior: reads remaining stdout, parses each non-empty line as s5cmd JSON, counts successes and
        # failures, sums object sizes, logs a summary.
        # Inputs: None. Outputs: True if no file failed else False.
        # Exceptions: JSON decode errors propagate for malformed output. Side effects: logging.
        self.get_current_stdout()
        s5cmd_results = [json.loads(i) for i in self.stdout.split("\n") if i]
        total_files = len([i for i in s5cmd_results if i["success"]])
        total_not_downloaded_files = len([i for i in s5cmd_results if not i["success"]])
        if total_not_downloaded_files == 0:
            all_upload = "all files"
            success = True
        else:
            all_upload = "not all files"
            success = False
        total_size = sum(i["object"]["size"] for i in s5cmd_results if "size" in i["object"])
        total_time = (datetime.now() - self.start_time).total_seconds()
        self._warning(
            f"[S3] Successfully downloaded {total_files} files for a total of {human_format(total_size)}B in {total_time}"
            f"sec ({all_upload}) from S3 at {self.s3_path} to {self.local_path}"
            f"(direct link: {self.s3_path_direct_link})"
        )
        return success

    def _start_uploading(
        self,
    ) -> (bool, str):
        # [FUNCTION DOC]
        # HELPER: begins the UPLOADING phase for full_local_path.
        # Behavior: globs files under full_local_path, acquires a lock on the first one (returns False if
        # locked), returns False with warning if the directory does not exist, then builds an s5cmd cp
        # command excluding lock files and launches it, setting state UPLOADING.
        # Inputs: None (uses full_local_path/full_s3_path). Outputs: bool success of starting.
        # Exceptions: IndexError if no files are found.
        # Side effects: file lock; spawns process; logging.
        # KNOWN ISSUE: the existence check happens after indexing the first globbed file, so an empty or missing directory raises IndexError instead of warning; return annotation says pair but returns bool.
        # Get a file lock on the first file
        local_files = glob.glob(self.full_local_path + "/**/*.*", recursive=True)

        locked = self._acquire_lock(local_files[0])
        if not locked:
            return False

        if not os.path.exists(self.full_local_path):
            message = f"[S3] Checkpoint {self.full_local_path} does not exist, cannot upload to S3"
            self._warning(message)
            return False

        self._warning(
            f"[S3] Uploading checkpoint in background from {self.full_local_path} to {self.full_s3_path} (direct link: {self.s3_path_direct_link})"
        )
        cmd = [self.s5cmd_path, "--json"]
        if self.s5cmd_credentials is not None:
            cmd += ["--credentials-file", self.s5cmd_credentials]
        if self.s5cmd_numworkers is not None:
            cmd += ["--numworkers", str(self.s5cmd_numworkers)]
        cmd += ["cp", "--exclude", "*.lock", "--exclude", "*.lock.*"]
        if self.s5cmd_concurrency is not None:
            cmd += ["--concurrency", str(self.s5cmd_concurrency)]
        cmd += [self.full_local_path, self.full_s3_path]

        self.process = self._popen(cmd)
        self.state = self.S3MoverState.UPLOADING

        return True

    def _post_uploading(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: summarizes a finished upload.
        # Behavior: parses s5cmd JSON lines, counts successes into num_uploaded_files, compares with the number
        # of local files having an extension, sums sizes, logs, calls post_upload_callback with the parsed
        # results (storing its output) if set, and releases the lock.
        # Inputs: None. Outputs: bool, True if counts match.
        # Exceptions: JSON errors or callback exceptions propagate. Side effects: logging, callback, lock release.
        self.get_current_stdout()
        s5cmd_results = [json.loads(i) for i in self.stdout.split("\n") if i]
        local_files = glob.glob(self.full_local_path + "/**/*.?*", recursive=True)
        total_files = len([i for i in s5cmd_results if i["success"]])
        self.num_uploaded_files = total_files
        if len(local_files) == total_files:
            all_upload = "all files"
            success = True
        else:
            all_upload = f"not all files: {len(local_files)} out of {total_files}"
            success = False
        total_size = sum(i["object"]["size"] for i in s5cmd_results if "size" in i["object"])
        total_time = (datetime.now() - self.start_time).total_seconds()
        self._warning(
            f"[S3] Successfully uploaded {total_files} files for a total of {human_format(total_size)}B in {total_time} sec"
            f"({all_upload}) from {self.full_local_path} to S3 at {self.full_s3_path} "
            f"(direct link: {self.s3_path_direct_link})"
        )
        if self.post_upload_callback:
            self.post_upload_callback_outputs = self.post_upload_callback(uploaded_files=s5cmd_results)
        self._release_lock()
        return success

    def _start_removing(self) -> (bool, str):
        # [FUNCTION DOC]
        # HELPER: begins the REMOVING_CHECKPOINT phase.
        # Behavior: lists top-level directories in local_path; if none, launches a no-op echo and sets state
        # REMOVING_CHECKPOINT; otherwise warns, locks the first directory (returns False if locked), launches
        # recursive verbose removal of all those directories and sets state REMOVING_CHECKPOINT.
        # Inputs: None. Outputs: bool.
        # Exceptions: None explicitly raised. Side effects: deletes local files via subprocess; lock; logging.
        top_dir_in_local_checkpoint = [dir for dir in glob.glob(self.local_path + "/*") if os.path.isdir(dir)]
        names_dir = [os.path.basename(dir) for dir in top_dir_in_local_checkpoint]
        if len(names_dir) == 0:
            # If the local is already empty or if we have already started duplicating in another process we skip with a noop
            self._warning("[S3] Local checkpoint empty. skipping removal")
            cmd = ["echo", "'skipping'"]
            self.process = self._popen(cmd)
            self.state = self.S3MoverState.REMOVING_CHECKPOINT
            return True

        self._warning(f"[S3] Removing checkpoint in background: {names_dir}")
        locked = self._acquire_lock(top_dir_in_local_checkpoint[0])
        if not locked:
            return False
        cmd = ["rm", "-rfv"] + top_dir_in_local_checkpoint
        self.process = self._popen(cmd)
        self.state = self.S3MoverState.REMOVING_CHECKPOINT
        return True

    def _post_removing(self) -> bool:
        # [FUNCTION DOC]
        # HELPER: summarizes a finished removal.
        # Behavior: counts removed file lines from rm output (excluding directory and lock lines), compares with
        # num_uploaded_files, releases the lock and logs.
        # Inputs: None. Outputs: bool, True if counts match.
        # Exceptions: None explicitly raised. Side effects: lock release; logging.
        self.get_current_stdout()
        local_files = [
            loc_f
            for loc_f in self.stdout.split("\n")
            if "directory" not in loc_f.lower() and loc_f and ".lock" not in loc_f
        ]
        if len(local_files) == self.num_uploaded_files:
            all_removed = "all files"
            success = True
        else:
            all_removed = "not all files"
            success = False
        self._release_lock()
        total_time = (datetime.now() - self.start_time).total_seconds()
        self._warning(
            f"[S3] Successfully removed {len(local_files)} local files ({all_removed}) from {self.local_path} (uploaded to {self.s3_path_direct_link}) in {total_time}"
        )
        return success

    def update(self) -> (str, str):
        # [FUNCTION DOC]
        # Behavior: polls the state machine. No process: reset to IDLE and return. Still running: return the
        # current state. Non-zero exit: read stdout, warn and return without changing state. Successful exit:
        # DOWNLOADING runs post-download and resets; UPLOADING runs post-upload then starts removal if
        # remove_after_upload else resets; REMOVING_CHECKPOINT runs post-removal and resets.
        # Inputs: None.
        # Outputs: pair of (state, stdout string).
        # Exceptions: None explicitly raised.
        # Side effects: may spawn processes, release locks, log.
        # KNOWN ISSUE: return type is inconsistent: the early returns give the enum member while the final return gives its string value; docstring lists states (REMOVING_DUPLICATED, DUPLICATING) that do not exist; a failed process leaves the mover stuck in its non-IDLE state with no recovery path.
        """Update the state of the mover: UPLOADING => REMOVING_DUPLICATED => DUPLICATING => REMOVING_CHECKPOINT => IDLE

        Returns:
            (str, str): The state and the stdout of the process if any
        """
        if self.process is None:
            self._reset_state()
            return self.state, self.stdout

        return_code = self.process.poll()
        if return_code is None:
            # Still running
            return self.state, self.stdout
        if return_code != 0:
            self.get_current_stdout()
            self._warning(
                f"[S3] Error running command {self.cmd} during process {self.state.value}, "
                f"return code {return_code}, return message {self.stdout}"
            )
            return self.state, self.stdout
        if self.state == self.S3MoverState.DOWNLOADING:
            self._post_downloading()
            self._reset_state()
        elif self.state == self.S3MoverState.UPLOADING:
            self._post_uploading()
            if self.remove_after_upload:
                self._start_removing()
            else:
                self._reset_state()
        elif self.state == self.S3MoverState.REMOVING_CHECKPOINT:
            self._post_removing()
            self._reset_state()

        return self.state.value, self.stdout

    def start_uploading(self, sub_folder=None):
        # [FUNCTION DOC]
        # Behavior: updates; if not IDLE warns and returns False; otherwise sets full_local_path and
        # full_s3_path (appending the sub folder when given) and starts the upload.
        # Inputs: sub_folder (optional str, e.g. the checkpoint step directory).
        # Outputs: bool, whether the upload was started.
        # Exceptions: see _start_uploading. Side effects: spawns process; lock; logging.
        """Start uploading last saved checkpoint to S3 in the background.

        After running this method, you should call regularly `update` to update the
        state to duplicating and then removing.

        For a blocking upload, call `wait_for_completion` or `distributed_wait_for_completion` after calling this method.
        """
        self.update()
        if self.state != self.S3MoverState.IDLE:
            message = "[S3] Cannot move to S3 as the previous checkpoint has not been uploaded and removed"
            self._warning(message)
            return False
        self.full_local_path = self.local_path + (f"/{sub_folder}" if sub_folder else "")
        self.full_s3_path = self.s3_path + (f"/{sub_folder}" if sub_folder else "")
        return self._start_uploading()

    def start_downloading(self):
        # [FUNCTION DOC]
        # Behavior: updates; if not IDLE warns and returns False; otherwise starts downloading the whole s3_path.
        # Inputs: None. Outputs: bool. Exceptions: None explicitly raised. Side effects: spawns process; logging.
        """Start downloading a checkpoint from S3 in the background.

        After running this method, you should call regularly `update` to update the
        state.

        For a blocking download, call `wait_for_completion` or `distributed_wait_for_completion` after calling this method.
        """
        self.update()
        if self.state != self.S3MoverState.IDLE:
            message = f"[S3] Cannot download from S3 as the state is not IDLE but {self.state.value}"
            self._warning(message)
            return False
        return self._start_downloading()
