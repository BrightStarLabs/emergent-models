"""
Fitness function implementations for cellular automata.

This module provides fitness functions that evaluate decoded outputs
against expected targets.
"""

import numpy as np
from typing import Callable
import warnings

from ..core.base import FitnessFn


class AbsoluteDifferenceFitness(FitnessFn):
    """
    Fitness function for the absolute difference task.
    
    Evaluates how well outputs match the absolute difference function: f(x) = |x - y|
    
    Examples
    --------
    >>> fitness = AbsoluteDifferenceFitness()
    >>> inputs = np.array([2, 4, 6, 8, 10])
    >>> outputs = np.array([2, 4, 6, 8, 10])  # Perfect absolute difference
    >>> scores = fitness(outputs, inputs)
    >>> print(scores)  # [1.0, 1.0, 1.0, 1.0, 1.0]
    """
    
    def __init__(self, continuous: bool = False):
        """
        Initialize absolute difference fitness function.
        
        Parameters
        ----------
        continuous : bool, default=False
            If True, use continuous error-based fitness.
            If False, use binary correct/incorrect fitness.
        """
        self.continuous = continuous
    
    def __call__(self, outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Evaluate absolute difference fitness.
        (P, B) + (B, ) -> (P, )
        
        Parameters
        ----------
        outputs : np.ndarray
            Decoded output values (P, B)
        targets : np.ndarray
            Target values (B, )
            
        Returns
        -------
        np.ndarray
            Fitness score for each genome (P, )
        """
        if self.continuous:
            # Continuous fitness based on error
            errors = np.abs(outputs - targets[None, :]).mean(1) # (P, 1)
            target_range = max(np.max(targets), np.max(outputs)) - min(np.min(targets), np.min(outputs))
            if target_range > 0:
                fitness = 1.0 - (errors / target_range)
                fitness = np.clip(fitness, 0.0, 1.0)
            else:
                fitness = (outputs == targets[None, :]).astype(float).mean(1)
        else:
            # Binary fitness: 1.0 if correct, 0.0 if incorrect
            fitness = (outputs == targets[None, :]).astype(float).mean(1)
        
        return fitness.squeeze()


class SparsityPenalizedFitness(FitnessFn):
    """
    Fitness function with integrated sparsity penalty.

    Combines task fitness with a penalty for programme complexity,
    providing better integration of the LAMBDA_P hyperparameter.

    Examples
    --------
    >>> base_fitness = AbsoluteDifferenceFitness()
    >>> fitness = SparsityPenalizedFitness(base_fitness, lambda_p=0.01)
    >>>
    >>> # Fitness automatically applies sparsity penalty
    >>> scores = fitness(outputs, inputs, sparsities=sparsities)
    """

    def __init__(self, base_fitness: FitnessFn, lambda_p: float = 0.01):
        """
        Initialize sparsity-penalized fitness.

        Parameters
        ----------
        base_fitness : FitnessFn
            Base fitness function for the task
        lambda_p : float, default=0.01
            Sparsity penalty coefficient (LAMBDA_P hyperparameter)
        """
        self.base_fitness = base_fitness
        self.lambda_p = lambda_p

    def __call__(self, outputs: np.ndarray, targets: np.ndarray,
                 sparsities: np.ndarray = None) -> np.ndarray:
        """
        Evaluate fitness with sparsity penalty.
        (P, B) + (B, ) + (P, ) -> (P, )

        Parameters
        ----------
        outputs : np.ndarray
            Decoded output values (P, B)
        targets : np.ndarray
            Target values (B, )
        sparsities : np.ndarray, optional
            Programme sparsity values for each genome

        Returns
        -------
        np.ndarray
            Fitness scores with sparsity penalty applied (P, )
        """
        base_scores = self.base_fitness(outputs, targets)

        if sparsities is not None:
            penalty = self.lambda_p * sparsities
            return base_scores - penalty
        else:
            return base_scores


class ComplexityRewardFitness(FitnessFn):
    """
    Fitness function that rewards programme complexity to prevent all-zero programmes.

    This fitness function adds a small bonus for non-empty programmes to encourage
    evolutionary exploration and prevent convergence to trivial all-zero solutions.

    Examples
    --------
    >>> base_fitness = AbsoluteDifferenceFitness()
    >>> fitness = ComplexityRewardFitness(base_fitness, complexity_bonus=0.05)
    >>>
    >>> # Fitness automatically rewards non-zero programmes
    >>> scores = fitness(outputs, inputs, sparsities=sparsities)
    """

    def __init__(self, base_fitness: FitnessFn, complexity_bonus: float = 0.05):
        """
        Initialize complexity-rewarding fitness.

        Parameters
        ----------
        base_fitness : FitnessFn
            Base fitness function for the task
        complexity_bonus : float, default=0.05
            Bonus coefficient for programme complexity (sparsity)
        """
        self.base_fitness = base_fitness
        self.complexity_bonus = complexity_bonus

    def __call__(self, outputs: np.ndarray, targets: np.ndarray,
                 sparsities: np.ndarray = None) -> np.ndarray:
        """
        Evaluate fitness with complexity bonus.
        (P, B) + (B, ) + (P, ) -> (P, )

        Parameters
        ----------
        outputs : np.ndarray
            Decoded output values (P, B)
        targets : np.ndarray
            Target values (B, )
        sparsities : np.ndarray, optional
            Programme sparsity values for each genome

        Returns
        -------
        np.ndarray
            Fitness scores with complexity bonus applied (P, )
        """
        base_scores = self.base_fitness(outputs, targets)

        if sparsities is not None:
            # Add bonus for non-zero programmes
            complexity_bonus = self.complexity_bonus * sparsities
            return base_scores + complexity_bonus
        else:
            return base_scores