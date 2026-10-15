# [MODULE DOC]
# Purpose: test helper providing a per-test temporary directory under tests/.test_cache that is removed when the context object is garbage collected.
# Main contents: TestContext.
import shutil
import uuid
from functools import lru_cache
from pathlib import Path


class TestContext:
    # [CLASS DOC]
    # Purpose: owns a unique temporary directory for one test.
    # Key attributes: _random_string, a uuid1 string naming the directory; _root_dir, the path tests/.test_cache (created on construction).
    # Construction inputs: none.
    # Invariant: the directory path is stable for the object's lifetime.
    def __init__(self):
        # [FUNCTION DOC]
        # Behavior: generates a uuid1 string and ensures the shared cache root directory exists.
        # Inputs: none.
        # Outputs: None.
        # Exceptions: OSError if the directory cannot be created.
        # Side effects: creates tests/.test_cache on disk.
        self._random_string = str(uuid.uuid1())
        self._root_dir = Path(__file__).parent.parent / ".test_cache"
        self._root_dir.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=1)
    def get_auto_remove_tmp_dir(self):
        # [FUNCTION DOC]
        # Behavior: builds the path root/uuid, creates it if missing and returns it; memoized with an lru_cache of size 1.
        # Inputs: none besides self.
        # Outputs: pathlib Path of the temporary directory.
        # Exceptions: OSError on creation failure.
        # Side effects: creates the directory.
        # KNOWN ISSUE: lru_cache on a method with maxsize 1 is keyed by self and holds a strong reference to self, so with several TestContext instances alive the cache evicts and also keeps the last instance alive (delaying its __del__ cleanup).
        path = self._root_dir / self._random_string
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __del__(self):
        # [FUNCTION DOC]
        # Behavior: on garbage collection, gets (or recreates) the temporary directory and recursively deletes it.
        # Inputs: none besides self.
        # Outputs: None.
        # Exceptions: errors are raised inside a finalizer (printed and ignored by Python).
        # Side effects: deletes the directory tree from disk.
        path = self.get_auto_remove_tmp_dir()
        shutil.rmtree(path)
