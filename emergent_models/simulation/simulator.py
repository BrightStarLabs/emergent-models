"""
Simulator implementation for cellular automata.

This module provides the Simulator class that performs pure CA evolution
on pre-encoded tapes, independent of encoding/decoding logic.
"""

import numpy as np
from typing import Tuple
import numba as nb

# Numba is a hard dependency
NUMBA_AVAILABLE = True

from ..core.state import StateModel
from ..core.space_model import SpaceModel


class Simulator:
    """
    Pure CA evolution simulator.
    
    Performs cellular automaton evolution on pre-encoded tapes.
    The simulator is encoder-agnostic - it just applies CA rules
    to whatever tape it receives.
    
    Examples
    --------
    >>> from emergent_models.core import StateModel, Tape1D
    >>> state = StateModel([0,1,2,3])
    >>> space = Tape1D(200, radius=1)
    >>> sim = Simulator(state, space, max_steps=100, halt_thresh=0.5)
    >>> 
    >>> # Simulate pre-encoded tape
    >>> tape = np.array([1, 0, 2, 3, 3, 0, 0, 2, 0, ...])
    >>> rule_table = np.random.randint(0, 4, 64)
    >>> final_tape = sim.run(tape, rule_table)
    """
    
    def __init__(self, state: StateModel, space: SpaceModel,
                 max_steps: int, halt_thresh: float):
        """
        Initialize simulator.
        
        Parameters
        ----------
        state : StateModel
            State model defining symbols and constraints
        space : SpaceModel
            Space model defining topology
        max_steps : int
            Maximum simulation steps
        halt_thresh : float
            Halting threshold (fraction of blue cells)
        """
        self.state = state
        self.space = space
        self.max_steps = max_steps
        self.halt_thresh = halt_thresh
        
        # Build JIT-compiled step kernel
        self._step_kernel = self._build_kernel()
    
    def _build_kernel(self):
        """Build JIT-compiled step function for this space/state configuration."""
        neigh = self.space.neighborhood()
        n_states = self.state.n_states
        
        if isinstance(self.space, type(self.space)) and hasattr(self.space, 'length'):
            # 1D tape
            length = self.space.size_hint()
            

            @nb.njit(cache=True, inline='always')
            def step_1d(tape, rule_table):
                next_tape = np.zeros(length, dtype=np.uint8)
                
                # Boundary cells stay 0
                next_tape[0] = 0
                next_tape[length-1] = 0
                
                # Apply rules to interior cells
                for x in range(1, length - 1):
                    left = tape[x - 1]
                    center = tape[x]
                    right = tape[x + 1]
                    
                    # Lookup table index
                    idx = (left << 4) | (center << 2) | right
                    if idx < len(rule_table):
                        next_tape[x] = rule_table[idx]
                
                return next_tape
            
            return step_1d
        else:
            # Generic space (slower)
            def step_generic(tape, rule_table):
                # Placeholder for generic space handling
                return tape.copy()
            
            return step_generic
    
    def run(self, tape: np.ndarray, rule_table: np.ndarray) -> np.ndarray:
        """
        Run CA simulation on pre-encoded tape with proper EM-4/3 halting semantics.

        Parameters
        ----------
        tape : np.ndarray
            Initial CA tape (pre-encoded)
        rule_table : np.ndarray
            Rule lookup table

        Returns
        -------
        np.ndarray
            Final CA tape after simulation (frozen state if halted)
        """
        current_tape = tape.copy()

        for step in range(self.max_steps):
            # Apply one step of CA evolution
            next_tape = self._step_kernel(current_tape, rule_table)

            # Check halting condition on the NEW state (after evolution)
            if self._check_halt(next_tape):
                # Return the frozen state (the state when halting occurred)
                return next_tape

            current_tape = next_tape

        # If no halting occurred, return final state
        return current_tape
    
    def _check_halt(self, tape: np.ndarray) -> bool:
        """
        Check if simulation should halt.
        
        Halts when the fraction of blue cells (state 3) exceeds threshold.
        
        Parameters
        ----------
        tape : np.ndarray
            Current CA tape
            
        Returns
        -------
        bool
            True if should halt, False otherwise
        """
        if 3 not in self.state.symbols:
            return False  # No blue state, never halt
        
        live_count = np.count_nonzero(tape)
        if live_count == 0:
            return False
        
        blue_count = np.count_nonzero(tape == 3)
        blue_fraction = blue_count / live_count
        
        return blue_fraction >= self.halt_thresh
    
    def run_batch(self, tapes: np.ndarray, rule_tables: np.ndarray) -> np.ndarray:
        """
        Run batch simulation on multiple tapes. (P, B, T) + (P, TableSize) -> (P, B, T)
        
        Parameters
        ----------
        tapes : np.ndarray
            Batch of initial tapes, shape (pop_size, batch_size, tape_length)
        rule_tables : np.ndarray
            Batch of rule tables, shape (pop_size, table_size)
            
        Returns
        -------
        np.ndarray
            Batch of final tapes, shape (pop_size, batch_size, tape_length)
        """
        pop_size, batch_size, tape_length = tapes.shape
        final_tapes = np.zeros_like(tapes)
        
        for p in range(pop_size):
            for b in range(batch_size):
                final_tapes[p, b] = self.run(tapes[p, b], rule_tables[p])
        
        return final_tapes


if NUMBA_AVAILABLE:
    @nb.njit(parallel=True, cache=True)
    def simulate_batch_numba(tapes: np.ndarray, rule_tables: np.ndarray,
                            max_steps: int, halt_thresh: float) -> np.ndarray:
        """
        High-performance batch simulation using Numba.
        (P, B, T) + (P, TableSize) -> (P, B, T)
        
        Parameters
        ----------
        tapes : np.ndarray
            Batch of initial tapes, shape (pop_size, batch_size, tape_length)
        rule_tables : np.ndarray
            Batch of rule tables, shape (pop_size, table_size)
        max_steps : int
            Maximum simulation steps
        halt_thresh : float
            Halting threshold
            
        Returns
        -------
        np.ndarray
            Batch of final tapes
        """
        pop_size, batch_size, tape_length = tapes.shape
        final_tapes = np.zeros_like(tapes)
        
        for p in nb.prange(pop_size):
            rule_table = rule_tables[p]
            for b in range(batch_size):
                tape = tapes[p, b].copy()

                for step in range(max_steps):
                    next_tape = np.zeros(tape_length, dtype=np.uint8)

                    # Apply CA rules (1D, radius=1)
                    for x in range(1, tape_length - 1):
                        left = tape[x - 1]
                        center = tape[x]
                        right = tape[x + 1]
                        idx = (left << 4) | (center << 2) | right
                        if idx < len(rule_table):
                            next_tape[x] = rule_table[idx]

                    # Check halting condition on the NEW state (EM-4/3 semantics)
                    live_count = 0
                    blue_count = 0
                    for j in range(tape_length):
                        if next_tape[j] != 0:
                            live_count += 1
                            if next_tape[j] == 3:
                                blue_count += 1

                    if live_count > 0 and blue_count / live_count >= halt_thresh:
                        # Halting occurred - freeze this state
                        final_tapes[p, b] = next_tape
                        break

                    tape = next_tape
                else:
                    # No halting occurred - use final state
                    final_tapes[p, b] = tape
        
        return final_tapes


class BatchSimulator:
    """
    High-performance batch simulator for population evaluation.
    
    Optimized for evaluating large populations of genomes in parallel.
    """
    
    def __init__(self, state: StateModel, space: SpaceModel,
                 max_steps: int, halt_thresh: float):
        """Initialize batch simulator."""
        self.state = state
        self.space = space
        self.max_steps = max_steps
        self.halt_thresh = halt_thresh
    
    def simulate_population(self, tapes: np.ndarray, 
                          rule_tables: np.ndarray) -> np.ndarray:
        """
        Simulate entire population in parallel.
        (P, B, T) + (P, TableSize) -> (P, B, T)
        
        Parameters
        ----------
        tapes : np.ndarray
            Population of initial tapes, shape (pop_size, batch_size, tape_length)
        rule_tables : np.ndarray
            Population of rule tables, shape (pop_size, table_size)
            
        Returns
        -------
        np.ndarray
            Population of final tapes
        """
        return simulate_batch_numba(tapes, rule_tables, 
                                      self.max_steps, self.halt_thresh)
