"""
Rule and programme sanitization utilities for EM-4/3 cellular automata.

This module provides functions to sanitize CA rules and programmes to ensure
they follow the required constraints and immutable entries.
"""

import numpy as np


def lut_idx(left: int, center: int, right: int) -> int:
    """
    Convert (left, center, right) neighborhood to LUT index.
    
    Args:
        left: Left neighbor cell value (0-3)
        center: Center cell value (0-3)  
        right: Right neighbor cell value (0-3)
        
    Returns:
        LUT index for the neighborhood configuration
    """
    return (left << 4) | (center << 2) | right


# Immutable LUT entries that must be preserved for EM-4/3 system
_IMMUTABLE = {
    lut_idx(0, 0, 0): 0,  # Empty space stays empty
    lut_idx(1, 1, 1): 1,  # Red propagates
    lut_idx(2, 2, 2): 2,  # Red propagates
    lut_idx(3, 3, 3): 3,  # Blue propagates
    lut_idx(3, 3, 0): 3,  # Blue boundary
    lut_idx(0, 0, 3): 0,  # Blue boundary
    lut_idx(3, 0, 0): 0,  # Blue boundary
}


def sanitize_rule(rule: np.ndarray) -> np.ndarray:
    """
    Sanitize a CA rule by enforcing immutable LUT entries and clipping values.
    
    Args:
        rule: Rule array of length 64 with values 0-3
        
    Returns:
        Sanitized rule array with immutable entries enforced and values clipped to 0-3
    """
    rule = rule.astype(np.uint8, copy=True)
    
    # Enforce immutable LUT entries
    for k, v in _IMMUTABLE.items():
        rule[k] = v
    
    # Clip values to valid range 0-3
    rule[rule > 3] &= 3
    
    return rule


def sanitize_programme(prog: np.ndarray) -> np.ndarray:
    """
    Sanitize a programme by removing invalid cell values.
    
    Blue cells (value 3) are not allowed in programmes as they are reserved
    for separators and halting indicators.
    
    Args:
        prog: Programme array with cell values
        
    Returns:
        Sanitized programme array with blue cells removed
    """
    prog = prog.astype(np.uint8, copy=True)
    
    # No blue cells allowed in programmes - they're reserved for separators
    prog[prog == 3] = 0
    
    # Clip to valid range 0-3 (though 3 is removed above)
    prog[prog > 3] &= 3
    
    return prog


def get_immutable_entries() -> dict:
    """
    Get the dictionary of immutable LUT entries for EM-4/3 system.
    
    Returns:
        Dictionary mapping LUT indices to required values
    """
    return _IMMUTABLE.copy()


def validate_rule_array(rule: np.ndarray) -> None:
    """
    Validate that a rule array has the correct shape and value range.
    
    Args:
        rule: Rule array to validate
        
    Raises:
        ValueError: If rule array is invalid
    """
    if not isinstance(rule, np.ndarray):
        raise ValueError("Rule must be a numpy array")
    
    if rule.shape != (64,):
        raise ValueError(f"Rule must have shape (64,), got {rule.shape}")
    
    if not np.issubdtype(rule.dtype, np.integer):
        raise ValueError(f"Rule must have integer dtype, got {rule.dtype}")
    
    if np.any(rule < 0) or np.any(rule > 3):
        raise ValueError("Rule values must be in range 0-3")


def validate_programme_array(prog: np.ndarray) -> None:
    """
    Validate that a programme array has valid values.
    
    Args:
        prog: Programme array to validate
        
    Raises:
        ValueError: If programme array is invalid
    """
    if not isinstance(prog, np.ndarray):
        raise ValueError("Programme must be a numpy array")
    
    if len(prog.shape) != 1:
        raise ValueError(f"Programme must be 1-dimensional, got shape {prog.shape}")
    
    if not np.issubdtype(prog.dtype, np.integer):
        raise ValueError(f"Programme must have integer dtype, got {prog.dtype}")
    
    if np.any(prog < 0) or np.any(prog > 3):
        raise ValueError("Programme values must be in range 0-3")
    
    if np.any(prog == 3):
        raise ValueError("Programme cannot contain blue cells (value 3)")
