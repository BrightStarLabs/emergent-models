"""
Core state management for cellular automata.

This module provides the StateModel class that defines the symbol table
and immutable constraints for CA systems.
"""

from typing import Dict, List
import numpy as np


class StateModel:
    """
    Holds the symbol table & immutable constraints for a CA system.
    
    This is the foundation component that defines:
    - Available states/symbols (e.g., [0,1,2,3] for EM-4/3)
    - Immutable lookup table entries that must be preserved
    - State validation and constraints
    
    Examples
    --------
    >>> # EM-4/3 system with 4 states
    >>> state = StateModel([0,1,2,3], immutable={0: 0})
    >>> 
    >>> # Binary system
    >>> state = StateModel([0,1])
    """
    
    def __init__(self, symbols: List[int], immutable: Dict[int, int] = None):
        """
        Initialize state model.
        
        Parameters
        ----------
        symbols : List[int]
            Available states/symbols (e.g., [0,1,2,3])
        immutable : Dict[int, int], optional
            Lookup table entries that must be preserved.
            Maps LUT index -> required output state.
        """
        self.symbols = list(symbols)
        self.immutable = immutable or {}
        self.n_states = len(symbols)
        
        # Validate inputs
        if not symbols:
            raise ValueError("Must provide at least one symbol")
        
        if min(symbols) < 0:
            raise ValueError("Symbols must be non-negative")
        
        if len(set(symbols)) != len(symbols):
            raise ValueError("Symbols must be unique")
        
        # Validate immutable entries
        for lut_idx, state in self.immutable.items():
            if not isinstance(lut_idx, int) or lut_idx < 0:
                raise ValueError(f"Invalid LUT index: {lut_idx}")
            if state not in symbols:
                raise ValueError(f"Immutable state {state} not in symbols {symbols}")
    
    def validate_state(self, state: int) -> bool:
        """Check if a state value is valid."""
        return state in self.symbols
    
    def validate_states(self, states: np.ndarray) -> bool:
        """Check if all states in array are valid."""
        return np.all(np.isin(states, self.symbols))
    
    def sanitize_states(self, states: np.ndarray) -> np.ndarray:
        """
        Sanitize state array by clipping to valid range.
        
        Parameters
        ----------
        states : np.ndarray
            Array of state values to sanitize
            
        Returns
        -------
        np.ndarray
            Sanitized array with values clipped to valid symbols
        """
        states = states.astype(np.uint8, copy=True)
        
        # Clip to valid range (assumes symbols are contiguous from 0)
        max_state = max(self.symbols)
        states = np.clip(states, 0, max_state)
        
        return states
    
    def __repr__(self) -> str:
        return f"StateModel(symbols={self.symbols}, n_states={self.n_states})"
