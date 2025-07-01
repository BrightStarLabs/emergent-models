"""
Custom fitness functions with Numba-compiled versions for fused kernel support.

This module provides fitness functions that expose Numba-compiled versions
for use in the trainer's fused kernel while maintaining clean Python APIs.
"""

import numpy as np
import numba as nb
from ..core.base import FitnessFn


class SumStateFitness(FitnessFn):
    """
    Fitness function that returns the sum of all final state values.
    
    This fitness function sums all cell values in the final CA tape,
    useful for tasks where you want to maximize or minimize total activity.
    
    Examples
    --------
    >>> fitness = SumStateFitness()
    >>> # Will be used automatically in fused kernel
    """
    
    def __call__(self, outputs: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Python interface (not used in fused kernel)."""
        # This is for non-fused usage, the fused kernel uses _fitness_numba
        return np.array([np.sum(outputs)])
    
    @staticmethod
    @nb.njit(cache=True, inline='always')
    def _fitness_numba(outputs: np.ndarray, inputs: np.ndarray,
                      final_tapes: np.ndarray, prog_len: int) -> float:
        """Numba-compiled fitness function for fused kernel."""
        total_sum = 0.0
        for i in range(final_tapes.shape[0]):
            for j in range(final_tapes.shape[1]):
                total_sum += final_tapes[i, j]
        return total_sum


class ConditionalCellFitness(FitnessFn):
    """
    Fitness function that gives score 10 if 5th cell contains 2, otherwise 0.
    
    This demonstrates conditional fitness based on specific cell values
    in the final CA tape.
    
    Examples
    --------
    >>> fitness = ConditionalCellFitness(cell_index=4, target_value=2, reward=10.0)
    """
    
    def __init__(self, cell_index: int = 4, target_value: int = 2, reward: float = 10.0):
        """
        Initialize conditional cell fitness.
        
        Parameters
        ----------
        cell_index : int, default=4
            Index of cell to check (0-based)
        target_value : int, default=2
            Value that cell should contain
        reward : float, default=10.0
            Fitness score if condition is met
        """
        self.cell_index = cell_index
        self.target_value = target_value
        self.reward = reward
    
    def __call__(self, outputs: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Python interface (not used in fused kernel)."""
        # This would need access to final tapes, which isn't available here
        return np.array([0.0])
    
    @staticmethod
    @nb.njit(cache=True, inline='always')
    def _fitness_numba(outputs: np.ndarray, inputs: np.ndarray,
                      final_tapes: np.ndarray, prog_len: int) -> float:
        """Numba-compiled fitness function for fused kernel."""
        # Check first tape's 5th cell (hardcoded for now)
        if final_tapes.shape[1] > 4 and final_tapes[0, 4] == 2:
            return 10.0
        else:
            return 0.0


class SqrtDifferenceFitness(FitnessFn):
    """
    Fitness function using square-root of difference between output and targets.
    
    This provides a different error metric than absolute difference,
    giving less penalty for small errors and more for large errors.
    
    Examples
    --------
    >>> fitness = SqrtDifferenceFitness()
    """
    
    def __call__(self, outputs: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Python interface (not used in fused kernel)."""
        targets = 2 * inputs  # Assuming doubling task
        diff = np.abs(outputs - targets)
        return -np.sqrt(diff)
    
    @staticmethod
    @nb.njit(cache=True, inline='always')
    def _fitness_numba(outputs: np.ndarray, inputs: np.ndarray,
                      final_tapes: np.ndarray, prog_len: int) -> float:
        """Numba-compiled fitness function for fused kernel."""
        total_error = 0.0
        for i in range(len(inputs)):
            target = 2 * inputs[i]  # Doubling task
            diff = abs(outputs[i] - target)
            total_error += np.sqrt(diff)
        return -total_error / len(inputs)


class CustomMathFitness(FitnessFn):
    """
    Example of custom mathematical operations on outputs.
    
    This demonstrates how to implement arbitrary mathematical
    operations in a Numba-compiled fitness function.
    
    Examples
    --------
    >>> fitness = CustomMathFitness()
    """
    
    def __call__(self, outputs: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Python interface (not used in fused kernel)."""
        # Custom math: sum of squares minus mean
        return np.array([np.sum(outputs**2) - np.mean(outputs)])
    
    @staticmethod
    @nb.njit(cache=True, inline='always')
    def _fitness_numba(outputs: np.ndarray, inputs: np.ndarray,
                      final_tapes: np.ndarray, prog_len: int) -> float:
        """Numba-compiled fitness function for fused kernel."""
        # Custom math: sum of squares minus mean
        sum_squares = 0.0
        sum_values = 0.0
        for i in range(len(outputs)):
            sum_squares += outputs[i] * outputs[i]
            sum_values += outputs[i]
        mean_value = sum_values / len(outputs)
        return sum_squares - mean_value


class TapePatternFitness(FitnessFn):
    """
    Fitness function that looks for specific patterns in the final tape.
    
    This demonstrates how to analyze the entire final CA tape
    for complex pattern matching.
    
    Examples
    --------
    >>> fitness = TapePatternFitness()
    """
    
    def __call__(self, outputs: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """Python interface (not used in fused kernel)."""
        return np.array([0.0])  # Would need tape access
    
    @staticmethod
    @nb.njit(cache=True, inline='always')
    def _fitness_numba(outputs: np.ndarray, inputs: np.ndarray,
                      final_tapes: np.ndarray, prog_len: int) -> float:
        """Numba-compiled fitness function for fused kernel."""
        # Look for alternating pattern: 0,1,0,1,0,1...
        pattern_score = 0.0
        for i in range(final_tapes.shape[0]):  # For each input
            for j in range(min(10, final_tapes.shape[1] - 1)):  # Check first 10 cells
                expected = j % 2  # Alternating 0,1,0,1...
                if final_tapes[i, j] == expected:
                    pattern_score += 1.0
        return pattern_score
