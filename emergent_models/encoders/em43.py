"""
EM-4/3 encoder for cellular automata.

This module provides the Em43Encoder that implements the standard
EM-4/3 positional encoding scheme.
"""

import numpy as np

import numba as nb

from ..core.base import Encoder
from ..core.state import StateModel
from ..core.space_model import SpaceModel


class Em43Encoder(Encoder):
    """
    EM-4/3 positional encoder.
    
    Implements the standard EM-4/3 encoding scheme:
    [programme] BB 0^(input+1) R 0...
    
    Where:
    - programme: The CA programme
    - BB: Blue separators (state 3)
    - 0^(input+1): input+1 zeros
    - R: Red beacon (state 2)
    
    Examples
    --------
    >>> from emergent_models.core import StateModel, Tape1D
    >>> state = StateModel([0,1,2,3])
    >>> space = Tape1D(200, radius=1)
    >>> encoder = Em43Encoder(state, space)
    >>> 
    >>> programme = np.array([1, 0, 2], dtype=np.uint8)
    >>> tape = encoder.encode(programme, input_val=5)
    >>> # tape: [1, 0, 2, 3, 3, 0, 0, 0, 0, 0, 2, 0, ...]
    """
    
    def __init__(self, state: StateModel, space: SpaceModel):
        """
        Initialize EM-4/3 encoder.

        Parameters
        ----------
        state : StateModel
            Must have states [0,1,2,3] for EM-4/3
        space : SpaceModel
            Space model (typically Tape1D)
        """
        super().__init__(state, space)
        
        # Validate state model for EM-4/3
        if state.n_states != 4 or state.symbols != [0,1,2,3]:
            raise ValueError("EM-4/3 encoder requires StateModel([0,1,2,3])")
    
    def encode(self, programme: np.ndarray, inp: np.ndarray) -> np.ndarray:
        """
        Encode single programme and batch of inputs into initial CA tapes.

        This method follows the training paradigm where:
        - Input data: (B, ) - batch of inputs
        - Programme: (L,) - single programme
        - Encoded tapes: (P, B, T) - population of encoded tapes

        For training, this is called P times (once per program) to create
        the full population of encoded tapes (P, B, T).

        Parameters
        ----------
        programme : np.ndarray, shape (L,)
            Single programme array (should not contain blue cells)
            
        inp : np.ndarray, shape (B,)
            Batch of input values to encode positionally

        Returns
        -------
        np.ndarray, shape (P, B, T)
            Batch of initial CA tapes: [programme] BB 0^(inp+1) R 0...
        """
        window = self.space.size_hint()
        
        # Use Numba-compiled batch encoding
        return _encode_em43_numba(programme, inp, window)

    def encode_population(self, programmes: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Encode population of programmes with batch of inputs.

        Parameters
        ----------
        programmes : np.ndarray, shape (P, L)
            Population of programmes
        inputs : np.ndarray, shape (B,)
            Batch of inputs (same size as programmes)

        Returns
        -------
        np.ndarray, shape (P, B, T)
            Encoded tapes for each program-input pair
        """

        P, L = programmes.shape
        B = inputs.shape[0]
        window = self.space.size_hint()

        # Create output tensor (P, B, T)
        encoded_tapes = np.zeros((P, B, window), dtype=np.uint8)

        # Encode each program with all inputs
        for p in range(P):
            encoded_tapes[p] = self.encode(programmes[p], inputs)

        return encoded_tapes
    
    def decode(self, tape: np.ndarray, programme_length: int) -> np.ndarray:
        """
        Decode batch of final CA tapes to output values using EXACT EM-4/3 decoding.

        This method follows the training paradigm where decoding is done
        on individual tapes from the population.

        The EM-4/3 decoding formula is:
        output = rightmost_red_position - (programme_length + 3)

        Where +3 accounts for: separator (2 cells) + 1 zero before input beacon

        Parameters
        ----------
        tape : np.ndarray, shape (B, T)
            Batch of final CA tapes after simulation
        programme_length : int
            Length of the programme (needed for exact decoding)

        Returns
        -------
        np.ndarray, shape (B,)
            Decoded output values, or -1 if no valid beacon found
        """
        if tape.ndim != 2:
            raise ValueError(f"Expected 2D tape array, got {tape.ndim}D. Use decode_population() for batch decoding.")

        return _decode_em43_numba(tape, programme_length)

    def decode_population(self, tapes: np.ndarray, programme_length: int) -> np.ndarray:
        """
        Decode population of final CA tapes to output values.

        Parameters
        ----------
        tapes : np.ndarray, shape (P, B, T)
            Population of final CA tapes after simulation
        programme_length : int
            Length of the programme (needed for exact decoding)

        Returns
        -------
        np.ndarray, shape (P, B)
            Decoded output values for each tape
        """
        if tapes.ndim != 3:
            raise ValueError(f"Expected 3D tape array (P, B, T), got {tapes.ndim}D")

        P = tapes.shape[0]
        B = tapes.shape[1]
        outputs = np.zeros((P,B), dtype=np.int32)
        for p in range(P):
            outputs[p] = self.decode(tapes[p], programme_length)
        return outputs



@nb.njit(inline='always', cache=True)
def _encode_em43_numba(programme: np.ndarray, inp: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized EM-4/3 encoding for batch of inputs."""
    L = len(programme)
    B = inp.shape[0]
    tape = np.zeros((B, window), dtype=np.uint8)

    for b in range(B):
        # Write programme
        for i in range(L):
            tape[b,i] = programme[i]
        
        # Write separator (BB)
        if L < window:
            tape[b,L] = 3
        if L + 1 < window:
            tape[b,L + 1] = 3
        
        # Write input beacon
        beacon_pos = L + 2 + inp[b] + 1
        if beacon_pos < window:
            tape[b,beacon_pos] = 2

    return tape
    

@nb.njit(inline='always', cache=True)
def _decode_em43_numba(tape: np.ndarray, programme_length: int) -> np.ndarray:
    """
    Numba-optimized EXACT EM-4/3 decoding for batch of tapes.

    Formula: output = rightmost_red_position - (programme_length + 3)
    Where +3 = separator (2 cells) + 1 zero before input beacon
    """
    B, T = tape.shape
    outputs = np.zeros(B, dtype=np.int32)
    for b in range(B):
        # Find rightmost red beacon
        rightmost_red = -1
        for i in range(T - 1, -1, -1):
            if tape[b,i] == 2:
                rightmost_red = i
                break

        if rightmost_red == -1:
            outputs[b] = -1

        # EXACT EM-4/3 decoding formula
        output = rightmost_red - (programme_length + 3)
        outputs[b] = max(0, output)
    return outputs


