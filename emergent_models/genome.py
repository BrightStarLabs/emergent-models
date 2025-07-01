"""
Genome implementation for cellular automata.

This module provides the Genome class that combines RuleSet and Programme
into a complete CA specification.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .rules.ruleset import RuleSet
from .rules.programme import Programme


@dataclass
class Genome:
    """
    Complete CA specification: RuleSet + Programme.
    
    A genome represents a complete cellular automaton specification
    consisting of:
    - rule: RuleSet defining state transition rules
    - programme: Programme defining static code segment
    - fitness: Current fitness score (mutable)
    
    Examples
    --------
    >>> from emergent_models.core import StateModel
    >>> from emergent_models.rules import RuleSet, Programme
    >>> 
    >>> state = StateModel([0,1,2,3])
    >>> rule = RuleSet(np.random.randint(0, 4, 64), state)
    >>> prog = Programme(np.array([1, 0, 2]), state)
    >>> 
    >>> genome = Genome(rule, prog)
    >>> genome.fitness = 0.85
    """
    
    rule: RuleSet
    programme: Programme
    fitness: float = 0.0
    
    def __post_init__(self):
        """Validate genome after initialization."""
        if not isinstance(self.rule, RuleSet):
            raise TypeError("rule must be a RuleSet instance")
        
        if not isinstance(self.programme, Programme):
            raise TypeError("programme must be a Programme instance")
        
        # Ensure rule and programme use compatible state models
        if self.rule.state.symbols != self.programme.state.symbols:
            raise ValueError("Rule and programme must use compatible state models")
    
    def mutate(self, rule_mutation_rate: float = 0.03,
               programme_mutation_rate: float = 0.08,
               rng: Optional[np.random.Generator] = None) -> 'Genome':
        """
        Create a mutated copy of this genome.
        
        Parameters
        ----------
        rule_mutation_rate : float, default=0.03
            Mutation rate for rule table
        programme_mutation_rate : float, default=0.08
            Mutation rate for programme
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        Genome
            New mutated genome
        """
        new_rule = self.rule.mutate(rule_mutation_rate, rng)
        new_programme = self.programme.mutate(programme_mutation_rate, rng)
        
        return Genome(new_rule, new_programme, fitness=0.0)
    
    def crossover(self, other: 'Genome',
                  rng: Optional[np.random.Generator] = None) -> 'Genome':
        """
        Create offspring through crossover with another genome.
        
        Parameters
        ----------
        other : Genome
            Other parent genome
        rng : np.random.Generator, optional
            Random number generator
            
        Returns
        -------
        Genome
            New offspring genome
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Crossover rule and programme independently
        child_rule = self.rule.crossover(other.rule, rng)
        child_programme = self.programme.crossover(other.programme, rng)
        
        return Genome(child_rule, child_programme, fitness=0.0)
    
    def copy(self) -> 'Genome':
        """Create a deep copy of this genome."""
        return Genome(self.rule.copy(), self.programme.copy(), self.fitness)

    def extract_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast extraction of rule table and programme arrays for performance-critical code.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (rule_table, programme_code) as numpy arrays
        """
        return self.rule.table, self.programme.code

    def sparsity_penalty(self) -> float:
        """
        Calculate sparsity penalty for this genome.

        Returns
        -------
        float
            Sparsity penalty value
        """
        # Default sparsity penalty (can be overridden by fitness functions)
        return 0.01 * self.programme.sparsity()
    
    def sparsity_penalty(self, penalty_weight: float = 0.01) -> float:
        """
        Calculate sparsity penalty for this genome.
        
        Encourages simpler programmes by penalizing complexity.
        
        Parameters
        ----------
        penalty_weight : float, default=0.01
            Weight for sparsity penalty
            
        Returns
        -------
        float
            Sparsity penalty value
        """
        return penalty_weight * self.programme.sparsity()
    
    def __repr__(self) -> str:
        return (f"Genome(rule_size={len(self.rule.table)}, "
                f"prog_length={len(self.programme)}, "
                f"fitness={self.fitness:.4f})")


def create_random_genome(state_model, programme_length: int = 10,
                        programme_sparsity: float = 0.6,
                        rng: Optional[np.random.Generator] = None) -> Genome:
    """
    Create a random genome with specified parameters.
    
    Parameters
    ----------
    state_model : StateModel
        State model to use for rule and programme
    programme_length : int, default=10
        Length of programme
    programme_sparsity : float, default=0.6
        Target sparsity for programme
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    Genome
        Random genome
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Create random rule (size depends on state model)
    # For 3-cell neighborhood with n states: table size = n^3
    n_states = state_model.n_states
    table_size = n_states ** 3
    rule_table = rng.integers(0, n_states, table_size, dtype=np.uint8)
    rule = RuleSet(rule_table, state_model)
    
    # Create random programme
    from .rules.programme import create_random_programme
    programme = create_random_programme(programme_length, state_model, 
                                      programme_sparsity, rng)
    
    return Genome(rule, programme)


def create_em43_genome(rule_array: Optional[np.ndarray] = None,
                      programme_code: Optional[np.ndarray] = None,
                      programme_length: int = 10,
                      rng: Optional[np.random.Generator] = None) -> Genome:
    """
    Create an EM-4/3 genome with proper constraints.
    
    Parameters
    ----------
    rule_array : np.ndarray, optional
        Rule array of length 64. If None, creates random rule.
    programme_code : np.ndarray, optional
        Programme code. If None, creates random programme.
    programme_length : int, default=10
        Length for random programme (if programme_code is None)
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    Genome
        EM-4/3 genome with proper constraints
    """
    from .rules.ruleset import create_em43_ruleset
    from .rules.programme import create_random_programme
    from .core.state import StateModel
    
    # Create EM-4/3 rule
    rule = create_em43_ruleset(rule_array, rng)
    
    # Create programme
    if programme_code is not None:
        programme = Programme(programme_code, rule.state)
    else:
        programme = create_random_programme(programme_length, rule.state,
                                          sparsity=0.6, rng=rng)
    
    return Genome(rule, programme)
