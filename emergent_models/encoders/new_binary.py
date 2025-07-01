"""
Binary encoder for cellular automata.

This module provides the BinaryEncoder that converts integers to
binary representation using CA states.
"""

import numpy as np
try:
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    nb = None

from ..core.base import Encoder
from ..core.state import StateModel
from ..core.space_model import SpaceModel


class BinaryEncoder(Encoder):
    """
    Binary encoder for CA systems.
    
    Converts integers to binary representation where:
    - 0 bits -> empty state (0)
    - 1 bits -> input state (configurable, default 2)
    
    Encoding format: [programme] BB [binary_input] 0...
    
    Examples
    --------
    >>> from emergent_models.core import StateModel, Tape1D
    >>> state = StateModel([0,1,2,3])
    >>> space = Tape1D(200, radius=1)
    >>> encoder = BinaryEncoder(state, space, input_state=2, max_bits=8)
    >>> 
    >>> programme = np.array([1, 0, 2], dtype=np.uint8)
    >>> tape = encoder.encode(programme, input_val=9)  # 9 = 00001001 binary
    >>> # tape: [1, 0, 2, 3, 3, 0, 0, 0, 0, 2, 0, 0, 2, 0, ...]
    """
    
    def __init__(self, state: StateModel, space: SpaceModel, 
                 input_state: int = 2, max_bits: int = 8):
        """
        Initialize binary encoder.
        
        Parameters
        ----------
        state : StateModel
            State model defining available symbols
        space : SpaceModel
            Space model (typically Tape1D)
        input_state : int, default=2
            CA state to use for '1' bits (default 2 = red)
        max_bits : int, default=8
            Maximum number of bits for binary representation
        """
        super().__init__(state, space)
        
        if input_state not in state.symbols:
            raise ValueError(f"Input state {input_state} not in state symbols {state.symbols}")
        
        if max_bits <= 0:
            raise ValueError("max_bits must be positive")
        
        self.input_state = input_state
        self.max_bits = max_bits
    
    def encode(self, programme: np.ndarray, inp: int) -> np.ndarray:
        """
        Encode single programme and single input into initial CA tape.

        This method follows the training paradigm where:
        - Input data: (B, D1) - batch of inputs
        - Programs: (P, L) - population of programs
        - Encoded tapes: (P, T) - population of encoded tapes

        Parameters
        ----------
        programme : np.ndarray, shape (L,)
            Single programme array
        inp : int
            Single input value to encode as binary

        Returns
        -------
        np.ndarray, shape (T,)
            Single initial CA tape: [programme] BB [binary_input] 0...
        """
        if inp < 0:
            raise ValueError("Input must be non-negative")

        if inp >= 2**self.max_bits:
            raise ValueError(f"Input {inp} too large for {self.max_bits} bits")

        window = self.space.size_hint()
        L = len(programme)

        # Check if encoding fits in window
        required_size = L + 2 + self.max_bits  # prog + BB + binary
        if required_size > window:
            raise ValueError(f"Window size {window} too small for programme length {L} and {self.max_bits} bits")

        # Initialize tape
        tape = np.zeros(window, dtype=np.uint8)

        # Write programme
        tape[:L] = programme

        # Write separator (BB)
        if L < window:
            tape[L] = 3  # Blue
        if L + 1 < window:
            tape[L + 1] = 3  # Blue

        # Convert input to binary and encode
        binary_str = format(inp, f'0{self.max_bits}b')
        start_pos = L + 2

        for i, bit in enumerate(binary_str):
            pos = start_pos + i
            if pos < window:
                if bit == '1':
                    tape[pos] = self.input_state
                # '0' bits remain as 0 (empty)

        return tape
    
    def decode(self, tape: np.ndarray) -> int:
        """
        Decode single final CA tape to output value.

        This method follows the training paradigm where decoding is done
        on individual tapes from the population.

        Looks for binary pattern after the separator and converts
        back to integer.

        Parameters
        ----------
        tape : np.ndarray, shape (T,)
            Single final CA tape after simulation

        Returns
        -------
        int
            Decoded output value, or -1 if no valid pattern found
        """
        if tape.ndim != 1:
            raise ValueError(f"Expected 1D tape array, got {tape.ndim}D. Use decode_population() for batch decoding.")

        return self._decode_single_tape(tape)

    def _decode_single_tape(self, tape: np.ndarray) -> int:
        """Decode a single tape to output value."""
        # Find blue separator pattern
        blue_positions = np.where(tape == 3)[0]
        if len(blue_positions) < 2:
            return -1

        # Find consecutive blue cells (separator)
        separator_end = -1
        for i in range(len(blue_positions) - 1):
            if blue_positions[i+1] == blue_positions[i] + 1:
                separator_end = blue_positions[i+1]
                break

        if separator_end == -1:
            return -1

        # Extract binary pattern after separator
        start_pos = separator_end + 1
        end_pos = min(start_pos + self.max_bits, len(tape))

        if start_pos >= len(tape):
            return -1

        # Convert binary pattern to integer
        binary_value = 0
        for i in range(start_pos, end_pos):
            bit_position = i - start_pos
            if tape[i] == self.input_state:
                binary_value |= (1 << (self.max_bits - 1 - bit_position))

        return binary_value


if NUMBA_AVAILABLE:
    @nb.njit(inline='always', cache=True)
    def _encode_binary_numba(programme: np.ndarray, inp: int, window: int,
                            input_state: int, max_bits: int) -> np.ndarray:
        """Numba-optimized binary encoding."""
        L = len(programme)
        tape = np.zeros(window, dtype=np.uint8)
        
        # Write programme
        for i in range(L):
            tape[i] = programme[i]
        
        # Write separator (BB)
        if L < window:
            tape[L] = 3
        if L + 1 < window:
            tape[L + 1] = 3
        
        # Convert input to binary and encode
        start_pos = L + 2
        for i in range(max_bits):
            pos = start_pos + i
            if pos < window:
                bit_position = max_bits - 1 - i
                if (inp >> bit_position) & 1:
                    tape[pos] = input_state
        
        return tape
    
    @nb.njit(inline='always', cache=True)
    def _decode_binary_numba(tape: np.ndarray, programme_length: int,
                            input_state: int, max_bits: int) -> int:
        """Numba-optimized binary decoding."""
        # Find separator end
        separator_end = programme_length + 1
        start_pos = separator_end + 1
        
        if start_pos >= len(tape):
            return -1
        
        # Convert binary pattern to integer
        binary_value = 0
        for i in range(max_bits):
            pos = start_pos + i
            if pos < len(tape) and tape[pos] == input_state:
                bit_position = max_bits - 1 - i
                binary_value |= (1 << bit_position)
        
        return binary_value


class BinaryEncoderNumba(BinaryEncoder):
    """Numba-optimized version of binary encoder."""
    
    def __init__(self, state: StateModel, space: SpaceModel,
                 input_state: int = 2, max_bits: int = 8):
        """Initialize Numba-optimized binary encoder."""
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba is required for BinaryEncoderNumba")
        super().__init__(state, space, input_state, max_bits)
    
    def encode(self, programme: np.ndarray, inp: int) -> np.ndarray:
        """Encode using Numba-compiled function."""
        window = self.space.size_hint()
        return _encode_binary_numba(programme, inp, window, 
                                   self.input_state, self.max_bits)
    
    def decode(self, tape: np.ndarray) -> int:
        """Decode using Numba-compiled function."""
        # For proper decoding, we need programme length
        # This is a limitation of the current interface
        return _decode_binary_numba(tape, 10, self.input_state, self.max_bits)
