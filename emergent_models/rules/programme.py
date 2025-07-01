"""
Programme implementation for cellular automata.

This module provides the Programme class that represents static
code segments with state model validation.
"""

import numpy as np
from typing import Optional

from ..core.state import StateModel


class Programme:
    """
    Static code segment for CA systems.
    
    Represents a programme (static code) that gets placed at the
    beginning of CA tapes. Validates against state model constraints.
    
    Examples
    --------
    >>> from emergent_models.core import StateModel
    >>> state = StateModel([0,1,2,3])
    >>> 
    >>> # Create programme from array
    >>> code = np.array([1, 0, 2, 1], dtype=np.uint8)
    >>> programme = Programme(code, state)
    >>> 
    >>> # Programme is automatically sanitized
    >>> print(programme.code)  # No blue cells (3) allowed
    """
    
    def __init__(self, code: np.ndarray, state: StateModel):
        """
        Initialize programme with code array.
        
        Parameters
        ----------
        code : np.ndarray
            Programme code array
        state : StateModel
            State model for validation and constraints
        """
        self.state = state
        
        # Validate and sanitize code
        code = np.asarray(code, dtype=np.uint8)
        if code.ndim != 1:
            raise ValueError("Programme code must be 1-dimensional")
        
        self.code = self._sanitize_code(code)
        self.length = len(self.code)
    
    def _sanitize_code(self, code: np.ndarray) -> np.ndarray:
        """
        Sanitize programme code according to state model.
        
        For EM-4/3 systems, blue cells (state 3) are not allowed
        in programmes as they are reserved for separators.
        
        Parameters
        ----------
        code : np.ndarray
            Raw programme code
            
        Returns
        -------
        np.ndarray
            Sanitized programme code
        """
        code = code.copy()
        
        # Apply state model sanitization
        code = self.state.sanitize_states(code)
        
        # For EM-4/3: remove blue cells (reserved for separators)
        if 3 in self.state.symbols:
            code[code == 3] = 0  # Replace blue with empty
        
        return code
    
    def mutate(self, mutation_rate: float = 0.08,
               rng: Optional[np.random.Generator] = None) -> 'Programme':
        """
        Create a mutated copy of this programme.
        
        Parameters
        ----------
        mutation_rate : float, default=0.08
            Probability of mutating each code position
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        Programme
            New mutated programme
        """
        if rng is None:
            rng = np.random.default_rng()
        
        new_code = self.code.copy()
        
        # Apply mutations
        mask = rng.random(len(new_code)) < mutation_rate
        if mask.any():
            # Choose from allowed states (exclude blue for EM-4/3)
            allowed_states = [s for s in self.state.symbols if s != 3]
            if not allowed_states:
                allowed_states = self.state.symbols
            
            # Balanced weights to avoid excessive zero bias
            if 0 in allowed_states and len(allowed_states) > 1:
                # Reduce zero bias from 70% to 40% for better diversity
                weights = [0.4 if s == 0 else 0.6/(len(allowed_states)-1)
                          for s in allowed_states]
            else:
                weights = None
            
            new_states = rng.choice(allowed_states, size=mask.sum(), p=weights)
            new_code[mask] = new_states
        
        return Programme(new_code, self.state)
    
    def crossover(self, other: 'Programme',
                  rng: Optional[np.random.Generator] = None) -> 'Programme':
        """
        Create offspring through crossover with another programme.
        
        Parameters
        ----------
        other : Programme
            Other parent programme
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        Programme
            New offspring programme
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Handle different lengths by using minimum length
        min_length = min(len(self.code), len(other.code))
        if min_length == 0:
            # One programme is empty, return copy of non-empty one
            return self.copy() if len(self.code) > 0 else other.copy()
        
        # Single-point crossover
        cut_point = rng.integers(1, min_length)
        
        # Create child code with same length as self
        child_code = np.zeros(len(self.code), dtype=np.uint8)
        child_code[:cut_point] = self.code[:cut_point]
        
        # Fill remaining positions from other parent (or zeros if other is shorter)
        remaining_length = len(self.code) - cut_point
        if cut_point < len(other.code):
            available_length = min(remaining_length, len(other.code) - cut_point)
            child_code[cut_point:cut_point + available_length] = \
                other.code[cut_point:cut_point + available_length]
        
        return Programme(child_code, self.state)
    
    def copy(self) -> 'Programme':
        """Create a deep copy of this programme."""
        return Programme(self.code.copy(), self.state)
    
    def sparsity(self) -> float:
        """
        Calculate programme sparsity (fraction of non-zero cells).
        
        Returns
        -------
        float
            Sparsity value between 0 and 1
        """
        if len(self.code) == 0:
            return 0.0
        return np.count_nonzero(self.code) / len(self.code)
    
    def __len__(self) -> int:
        """Get programme length."""
        return self.length
    
    def __getitem__(self, key):
        """Get programme code elements."""
        return self.code[key]
    
    def __repr__(self) -> str:
        return f"Programme(length={self.length}, sparsity={self.sparsity():.3f})"


def create_random_programme(length: int, state: StateModel,
                           sparsity: float = 0.6,
                           rng: Optional[np.random.Generator] = None) -> Programme:
    """
    Create a random programme with specified sparsity.
    
    Parameters
    ----------
    length : int
        Programme length
    state : StateModel
        State model for validation
    sparsity : float, default=0.6
        Target sparsity (fraction of non-zero cells)
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    Programme
        Random programme
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if not 0 <= sparsity <= 1:
        raise ValueError("Sparsity must be between 0 and 1")
    
    # Create code array
    code = np.zeros(length, dtype=np.uint8)
    
    # Determine number of non-zero cells
    n_nonzero = int(length * sparsity)
    
    if n_nonzero > 0:
        # Choose positions for non-zero cells
        positions = rng.choice(length, size=n_nonzero, replace=False)
        
        # Choose states (exclude blue for EM-4/3)
        allowed_states = [s for s in state.symbols if s != 3 and s != 0]
        if not allowed_states:
            allowed_states = [s for s in state.symbols if s != 0]
        
        if allowed_states:
            states = rng.choice(allowed_states, size=n_nonzero)
            code[positions] = states
    
    return Programme(code, state)
