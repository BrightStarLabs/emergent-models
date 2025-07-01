"""
RuleSet implementation for cellular automata.

This module provides the RuleSet class that defines state transition
rules using lookup tables with state model constraints.
"""

import numpy as np
from typing import Optional

from ..core.state import StateModel


class RuleSet:
    """
    CA rule lookup table with state model constraints.
    
    Represents cellular automaton rules as a lookup table that maps
    neighborhood configurations to output states. Enforces immutable
    constraints from the state model.
    
    Examples
    --------
    >>> from emergent_models.core import StateModel
    >>> state = StateModel([0,1,2,3], immutable={0: 0})
    >>> 
    >>> # Random rule table
    >>> rule_table = np.random.randint(0, 4, 64, dtype=np.uint8)
    >>> ruleset = RuleSet(rule_table, state)
    >>> 
    >>> # Rule table is automatically sanitized
    >>> assert ruleset.table[0] == 0  # Immutable constraint enforced
    """
    
    def __init__(self, table: np.ndarray, state: StateModel):
        """
        Initialize rule set with lookup table.
        
        Parameters
        ----------
        table : np.ndarray
            Lookup table mapping neighborhood indices to output states.
            For 3-cell neighborhood with 4 states, should be shape (64,).
        state : StateModel
            State model defining symbols and constraints
        """
        self.state = state
        
        # Validate and sanitize table
        table = np.asarray(table, dtype=np.uint8)
        if table.ndim != 1:
            raise ValueError("Rule table must be 1-dimensional")
        
        # Apply state model constraints
        self.table = self._sanitize_table(table)
        
        # Cache neighborhood size (assumes 3-cell for now)
        self.neighborhood_size = 3
    
    def _sanitize_table(self, table: np.ndarray) -> np.ndarray:
        """
        Sanitize rule table by applying state model constraints.
        
        Parameters
        ----------
        table : np.ndarray
            Raw rule table
            
        Returns
        -------
        np.ndarray
            Sanitized rule table with constraints applied
        """
        table = table.copy()
        
        # Clip values to valid state range
        table = self.state.sanitize_states(table)
        
        # Apply immutable constraints
        for lut_idx, required_state in self.state.immutable.items():
            if lut_idx < len(table):
                table[lut_idx] = required_state
        
        return table
    
    def lookup(self, left: int, center: int, right: int) -> int:
        """
        Look up output state for neighborhood configuration.
        
        Parameters
        ----------
        left : int
            Left neighbor state
        center : int
            Center cell state
        right : int
            Right neighbor state
            
        Returns
        -------
        int
            Output state for this neighborhood
        """
        # Convert neighborhood to lookup table index
        # For 3-cell neighborhood: idx = left*16 + center*4 + right
        idx = (left << 4) | (center << 2) | right
        
        if idx >= len(self.table):
            return 0  # Default to empty state
        
        return self.table[idx]
    
    def mutate(self, mutation_rate: float = 0.03, 
               rng: Optional[np.random.Generator] = None) -> 'RuleSet':
        """
        Create a mutated copy of this rule set.
        
        Parameters
        ----------
        mutation_rate : float, default=0.03
            Probability of mutating each table entry
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        RuleSet
            New mutated rule set
        """
        if rng is None:
            rng = np.random.default_rng()
        
        new_table = self.table.copy()
        
        # Apply mutations
        mask = rng.random(len(new_table)) < mutation_rate
        if mask.any():
            # Choose random states from available symbols
            new_states = rng.choice(self.state.symbols, size=mask.sum())
            new_table[mask] = new_states
        
        return RuleSet(new_table, self.state)
    
    def crossover(self, other: 'RuleSet', 
                  rng: Optional[np.random.Generator] = None) -> 'RuleSet':
        """
        Create offspring through crossover with another rule set.
        
        Parameters
        ----------
        other : RuleSet
            Other parent rule set
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        RuleSet
            New offspring rule set
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if len(self.table) != len(other.table):
            raise ValueError("Rule tables must have same length for crossover")
        
        # Single-point crossover
        cut_point = rng.integers(1, len(self.table))
        child_table = np.concatenate([
            self.table[:cut_point],
            other.table[cut_point:]
        ])
        
        return RuleSet(child_table, self.state)
    
    def copy(self) -> 'RuleSet':
        """Create a deep copy of this rule set."""
        return RuleSet(self.table.copy(), self.state)
    
    def __repr__(self) -> str:
        return f"RuleSet(table_size={len(self.table)}, state={self.state})"


def create_em43_ruleset(rule_array: Optional[np.ndarray] = None,
                       rng: Optional[np.random.Generator] = None) -> RuleSet:
    """
    Create an EM-4/3 rule set with proper constraints.
    
    Parameters
    ----------
    rule_array : np.ndarray, optional
        Rule array of length 64. If None, creates random rule.
    rng : np.random.Generator, optional
        Random number generator for random rule creation
        
    Returns
    -------
    RuleSet
        EM-4/3 rule set with immutable constraints
    """
    # EM-4/3 state model with immutable constraints
    immutable = {
        0: 0,   # Empty space stays empty (000 -> 0)
        21: 1,  # Red propagates (111 -> 1)
        42: 2,  # Red propagates (222 -> 2)
        63: 3,  # Blue propagates (333 -> 3)
        60: 3,  # Blue boundary (330 -> 3)
        3: 0,   # Blue boundary (003 -> 0)
        48: 0,  # Blue boundary (300 -> 0)
    }
    
    state = StateModel([0, 1, 2, 3], immutable=immutable)
    
    if rule_array is None:
        if rng is None:
            rng = np.random.default_rng()
        rule_array = rng.integers(0, 4, 64, dtype=np.uint8)
    
    return RuleSet(rule_array, state)


def create_elementary_ruleset(rule_number: int) -> RuleSet:
    """
    Create an elementary CA rule set.
    
    Parameters
    ----------
    rule_number : int
        Elementary CA rule number (0-255)
        
    Returns
    -------
    RuleSet
        Elementary CA rule set
    """
    if not 0 <= rule_number <= 255:
        raise ValueError("Elementary CA rule number must be 0-255")
    
    # Binary state model (no immutable constraints)
    state = StateModel([0, 1])
    
    # Convert rule number to lookup table
    table = np.zeros(8, dtype=np.uint8)
    for i in range(8):
        table[i] = (rule_number >> i) & 1
    
    return RuleSet(table, state)
