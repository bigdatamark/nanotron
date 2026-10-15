# [MODULE DOC]
# Purpose: test-support helper for the Llama example tests. Provides set_system_path, which makes the
# nanotron repository root and the examples/llama directory importable so tests can import
# "examples.llama..." and "tests.helpers..." modules.
# Contents: set_system_path. No import-time side effects beyond importing nanotron lazily inside the function.
import importlib
import sys
from pathlib import Path


def set_system_path():
    # [FUNCTION DOC]
    # HELPER: prepares sys.path for the llama example tests.
    # Behavior: imports the nanotron package, takes the directory three levels above its __init__ file
    # (src/nanotron -> src -> repo root) and inserts it at the front of sys.path; then inserts the
    # examples/llama directory (parent of this tests folder) at the front as well, so it ends up first.
    # Inputs: none. Outputs: None. Exceptions: ImportError if nanotron is not installed.
    # Side effects: mutates the global sys.path (two front insertions).
    package = importlib.import_module("nanotron")
    # NOTE:  Path(package.__file__).parent = .../nanotron/src/nanotron
    # we want .../nanotron
    package_path = Path(package.__file__).parent.parent.parent
    sys.path.insert(0, str(package_path))

    # we also want ../llama
    llama_path = Path(__file__).parent.parent
    sys.path.insert(0, str(llama_path))
