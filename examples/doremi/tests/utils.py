# [MODULE DOC]
# Purpose: Shared helpers for the DoReMi example tests (imported as a top-level "utils" module when pytest runs from the tests folder).
# Main contents: set_system_path, create_dummy_dataset. Side effects at import: imports the datasets library.
import importlib
import sys
from pathlib import Path

from datasets import Dataset


def set_system_path():
    # [FUNCTION DOC]
    # HELPER: makes the repository root importable so tests can import examples.doremi.
    # Behavior: Imports nanotron, takes the directory three levels above its __init__ file (the repo root for a src layout), and appends it to sys.path.
    # Inputs: None. Outputs: None.
    # Exceptions: ImportError if nanotron is not installed.
    # Side effects: mutates sys.path (appends on every call).
    package = importlib.import_module("nanotron")
    # NOTE:  Path(package.__file__).parent = .../nanotron/src/nanotron
    # we want .../nanotron
    package_path = Path(package.__file__).parent.parent.parent
    sys.path.append(str(package_path))


def create_dummy_dataset(num_items: int):
    # [FUNCTION DOC]
    # HELPER: builds a tiny HF dataset for tests.
    # Behavior: Returns a datasets.Dataset with one column "text" holding the integers 0 to num_items-1.
    # Inputs: num_items: int. Outputs: datasets.Dataset. Exceptions: None explicitly raised. Side effects: None.
    data = {"text": list(range(num_items))}
    return Dataset.from_dict(data)
