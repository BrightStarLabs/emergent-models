"""
Base interfaces and common utilities for the component-first architecture.

This module provides the foundational abstract base classes and utilities
that other components depend on.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from .state import StateModel
from .space_model import SpaceModel


class Encoder(ABC):
    """
    Abstract base class for encoding/decoding data to/from CA tapes.
    
    Encoders are responsible for converting between problem domain data
    (integers, strings, etc.) and CA tape representations. They are
    completely separate from simulation logic.
    """
    
    def __init__(self, state: StateModel, space: SpaceModel):
        """
        Initialize encoder with state and space models.
        
        Parameters
        ----------
        state : StateModel
            State model defining available symbols
        space : SpaceModel  
            Space model defining topology
        """
        self.state = state
        self.space = space
    
    @abstractmethod
    def encode(self, programme: np.ndarray, inp: int) -> np.ndarray:
        """
        Encode programme and input into initial CA tape.
        
        Parameters
        ----------
        programme : np.ndarray
            Programme array
        inp : int
            Input value to encode
            
        Returns
        -------
        np.ndarray
            Initial CA tape ready for simulation
        """
        pass
    
    @abstractmethod
    def decode(self, tape: np.ndarray) -> int:
        """
        Decode final CA tape to output value.
        
        Parameters
        ----------
        tape : np.ndarray
            Final CA tape after simulation
            
        Returns
        -------
        int
            Decoded output value
        """
        pass


class FitnessFn(ABC):
    """
    Abstract base class for fitness/loss functions.
    
    Fitness functions evaluate how well outputs match expected targets.
    They work on decoded outputs, not raw CA states.
    """
    
    @abstractmethod
    def __call__(self, outputs: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for a batch of outputs.
        
        Parameters
        ----------
        outputs : np.ndarray
            Decoded output values
        inputs : np.ndarray
            Input values
            
        Returns
        -------
        np.ndarray
            Fitness scores (higher is better)
        """
        pass


class Monitor(ABC):
    """
    Abstract base class for training monitoring/logging.
    
    Monitors handle side effects like logging, plotting, checkpointing.
    """
    
    @abstractmethod
    def update(self, generation: int, scores: np.ndarray, **kwargs) -> None:
        """
        Update monitor with training progress.
        
        Parameters
        ----------
        generation : int
            Current generation number
        scores : np.ndarray
            Fitness scores for current generation
        **kwargs
            Additional monitoring data
        """
        pass


class ConsoleLogger(Monitor):
    """Simple console logger for training progress."""
    
    def __init__(self, log_every: int = 10):
        """
        Initialize console logger.
        
        Parameters
        ----------
        log_every : int, default=10
            Log every N generations
        """
        self.log_every = log_every
    
    def update(self, generation: int, scores: np.ndarray, **kwargs) -> None:
        """Log training progress to console."""
        if generation % self.log_every == 0:
            best_score = np.max(scores)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            print(f"Gen {generation:4d}: "
                  f"Best={best_score:.4f}, "
                  f"Mean={mean_score:.4f}±{std_score:.4f}")
