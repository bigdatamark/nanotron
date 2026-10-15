# [MODULE DOC]
# Purpose: standalone manual script (at repo root, not under tests) exercising the nanotron_timer decorator from nanotron.logging.timers with default, explicit CUDA, CPU and custom-name timer variants.
# Main contents: four decorated test functions and a main block that runs them and logs all timers.
# Import-time side effects: enables the global nanotron_timer at import; decorating registers timers.
# Exports: none intended; pytest may collect the test_ functions.
"""Test script for the timer decorator with both CPU and CUDA timer types."""

from nanotron.logging.timers import nanotron_timer, TimerType
import time
import torch

# Enable timers for testing
nanotron_timer.enable()

# Test with default CUDA timing
@nanotron_timer
def test_default_decorator():
    # [FUNCTION DOC]
    # Behavior: decorated with nanotron_timer using defaults (CUDA timing). Sleeps 0.1 s; if CUDA is available, multiplies a random 1000x1000 GPU matrix by itself and synchronizes.
    # Inputs: none.
    # Outputs: the string "Done".
    # Exceptions: None explicitly raised.
    # Side effects: timer recording, GPU allocation/compute, sleeping.
    """Test function with default CUDA timing."""
    # Simulate some work
    time.sleep(0.1)
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
    return "Done"

# Test with explicit CUDA timing
@nanotron_timer(timer_type=TimerType.CUDA)
def test_cuda_decorator():
    # [FUNCTION DOC]
    # Behavior: same work as test_default_decorator but with the timer type explicitly set to CUDA.
    # Inputs: none.
    # Outputs: the string "Done".
    # Exceptions: None explicitly raised.
    # Side effects: timer recording, GPU work if available, sleeping.
    """Test function with explicit CUDA timing."""
    # Simulate some work
    time.sleep(0.1)
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
    return "Done"

# Test with CPU timing
@nanotron_timer(timer_type=TimerType.CPU)
def test_cpu_decorator():
    # [FUNCTION DOC]
    # Behavior: timed with a CPU timer; sleeps 0.2 s.
    # Inputs: none.
    # Outputs: the string "Done".
    # Exceptions: None explicitly raised.
    # Side effects: timer recording, sleeping.
    """Test function with CPU timing."""
    # Simulate some CPU work
    time.sleep(0.2)
    return "Done"

# Test with custom name
@nanotron_timer("custom_name")
def test_custom_name_decorator():
    # [FUNCTION DOC]
    # Behavior: timed under the custom timer name "custom_name"; sleeps 0.1 s.
    # Inputs: none.
    # Outputs: the string "Done".
    # Exceptions: None explicitly raised.
    # Side effects: timer recording, sleeping.
    """Test function with custom name."""
    # Simulate some work
    time.sleep(0.1)
    return "Done"

if __name__ == "__main__":
    print("Testing timer decorators...")
    
    # Run the test functions
    test_default_decorator()
    test_cuda_decorator()
    test_cpu_decorator()
    test_custom_name_decorator()
    
    # Log all timers
    print("\nTimer results:")
    nanotron_timer.log_all(rank=None)  # Log on all ranks
    
    print("\nTest completed successfully!")
