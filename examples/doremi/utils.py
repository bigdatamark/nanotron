# [MODULE DOC]
# Purpose: Formatting helper for the DoReMi example scripts.
# Main contents: print_array_for_human. Side effects at import: None.
from typing import List


def print_array_for_human(arr: List[float], precision: int = 5) -> str:
    # [FUNCTION DOC]
    # HELPER: formats domain weight lists for logging.
    # Behavior: Formats each number with the given number of decimals and joins them inside square brackets separated by commas.
    # Inputs: arr: list of floats (or iterable of numbers); precision: int, default 5.
    # Outputs: string.
    # Exceptions: None explicitly raised. Side effects: None (despite the name, it does not print).
    formatted_elements = [f"{x:.{precision}f}" for x in arr]
    return "[" + ", ".join(formatted_elements) + "]"
