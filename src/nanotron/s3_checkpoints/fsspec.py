# [MODULE DOC]
# Purpose: Thin wrappers over fsspec to open, copy and classify paths that may be local or remote.
# Main contents: get_filesystem_and_path, fs_open, fs_copy, check_path_is_local.
# Import-time side effects: None.
import contextlib
from pathlib import Path
from typing import Tuple, Union

import fsspec
from fsspec.implementations import local


def get_filesystem_and_path(path: Path, storage_options=None) -> Tuple[fsspec.AbstractFileSystem, str]:
    # [FUNCTION DOC]
    # HELPER: resolves a path into an fsspec filesystem plus path, used by the other functions.
    # Behavior: calls fsspec's get_fs_token_paths on the string form of path and expects exactly one path.
    # Inputs: path (Path or URL-like string), storage_options (optional dict for the filesystem).
    # Outputs: pair of (fsspec AbstractFileSystem, normalized path string).
    # Exceptions: AssertionError if the path expands to more or fewer than one path.
    # Side effects: None.
    # Use supported filesystems in `fsspec`. If you need another one, please use `fsspec.registry.register_implementation`
    # DO NOT USE `mode` argument as it adds a suffix `0.part` when using `mode="w"`.
    fs, _, paths = fsspec.core.get_fs_token_paths(str(path), storage_options=storage_options)
    assert len(paths) == 1
    return fs, paths[0]


@contextlib.contextmanager
def fs_open(
    file: Union[str, Path],
    mode="r",
):
    # [FUNCTION DOC]
    # Behavior: context manager that resolves the filesystem for file and yields the opened file object,
    # closing it on exit.
    # Inputs: file (str or Path, local or remote), mode (default "r").
    # Outputs: yields a file-like object.
    # Exceptions: propagates filesystem errors (e.g. file not found). Side effects: file I/O.
    # TODO @thomasw21: pass storage options.
    fs, path = get_filesystem_and_path(file)
    with fs.open(path, mode=mode) as f:
        yield f


def fs_copy(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
):
    # [FUNCTION DOC]
    # Behavior: reads the whole input file into memory in binary mode and writes it to the output file.
    # Inputs: input_file, output_file (str or Path, any supported filesystem).
    # Outputs: None. Exceptions: propagates I/O errors. Side effects: writes output_file.
    """Copy file from input to output (possibly on s3/other fs)"""
    with fs_open(input_file, mode="rb") as fi, fs_open(output_file, mode="wb") as fo:
        fo.write(fi.read())


def check_path_is_local(path: Path, storage_options=None) -> bool:
    # [FUNCTION DOC]
    # Behavior: returns whether the resolved filesystem for path is fsspec's LocalFileSystem.
    # Inputs: path, storage_options (optional). Outputs: bool.
    # Exceptions: AssertionError from get_filesystem_and_path. Side effects: None.
    return isinstance(get_filesystem_and_path(path=path, storage_options=storage_options)[0], local.LocalFileSystem)
