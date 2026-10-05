# =====================================================================
# FILE: utils/helpers.py
# =====================================================================
from typing import Callable


def safe_delay_time(delay_function: Callable[[], float]) -> float:
    """
    Ensure delay times are non-negative.
    
    Standalone helper function for delay time validation.
    
    Args:
        delay_function: Function returning delay time
        
    Returns:
        Non-negative delay time
    """
    delay = delay_function()
    return max(0.0, delay)