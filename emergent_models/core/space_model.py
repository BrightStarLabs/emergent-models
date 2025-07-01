"""
Space topology models for cellular automata.

This module provides SpaceModel classes that define the topology
and neighborhood structure for CA systems.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class SpaceModel(ABC):
    """
    Abstract base class for CA space topology & neighborhood.
    
    Defines the spatial structure and neighborhood relationships
    for cellular automata. This is separate from the actual data
    storage (which is handled by numpy arrays in the simulation).
    """
    
    @abstractmethod
    def neighborhood(self) -> Tuple[int, ...]:
        """
        Get relative indices for neighborhood.
        
        Returns
        -------
        Tuple[int, ...]
            Relative indices for neighborhood cells.
            E.g., (-1, 0, 1) for radius-1 1D neighborhood.
        """
        pass
    
    @abstractmethod
    def size_hint(self) -> int:
        """
        Get suggested size for this space type.
        
        Returns
        -------
        int
            Suggested size for allocating arrays.
        """
        pass


class Tape1D(SpaceModel):
    """
    1D tape space with radius-based neighborhood.
    
    Represents a 1-dimensional cellular automaton tape with
    configurable neighborhood radius.
    
    Examples
    --------
    >>> # Standard elementary CA (radius 1)
    >>> space = Tape1D(length=100, radius=1)
    >>> space.neighborhood()
    (-1, 0, 1)
    
    >>> # Larger neighborhood
    >>> space = Tape1D(length=200, radius=2) 
    >>> space.neighborhood()
    (-2, -1, 0, 1, 2)
    """
    
    def __init__(self, length: int, radius: int = 1):
        """
        Initialize 1D tape space.
        
        Parameters
        ----------
        length : int
            Length of the tape
        radius : int, default=1
            Neighborhood radius (number of cells on each side)
        """
        if length <= 0:
            raise ValueError("Length must be positive")
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        
        self.length = length
        self.radius = radius
    
    def neighborhood(self) -> Tuple[int, ...]:
        """Get relative indices for neighborhood."""
        return tuple(range(-self.radius, self.radius + 1))
    
    def size_hint(self) -> int:
        """Get the tape length."""
        return self.length
    
    def neighborhood_size(self) -> int:
        """Get the number of cells in neighborhood."""
        return 2 * self.radius + 1
    
    def __repr__(self) -> str:
        return f"Tape1D(length={self.length}, radius={self.radius})"


class Grid2D(SpaceModel):
    """
    2D grid space with Moore or von Neumann neighborhood.
    
    Represents a 2-dimensional cellular automaton grid.
    """
    
    def __init__(self, height: int, width: int, radius: int = 1, 
                 neighborhood_type: str = "moore"):
        """
        Initialize 2D grid space.
        
        Parameters
        ----------
        height : int
            Grid height
        width : int
            Grid width  
        radius : int, default=1
            Neighborhood radius
        neighborhood_type : str, default="moore"
            Type of neighborhood ("moore" or "von_neumann")
        """
        if height <= 0 or width <= 0:
            raise ValueError("Height and width must be positive")
        if radius < 0:
            raise ValueError("Radius must be non-negative")
        if neighborhood_type not in ["moore", "von_neumann"]:
            raise ValueError("Neighborhood type must be 'moore' or 'von_neumann'")
        
        self.height = height
        self.width = width
        self.radius = radius
        self.neighborhood_type = neighborhood_type
    
    def neighborhood(self) -> Tuple[int, ...]:
        """Get relative indices for neighborhood."""
        if self.neighborhood_type == "moore":
            # Moore neighborhood: all cells within radius
            indices = []
            for dy in range(-self.radius, self.radius + 1):
                for dx in range(-self.radius, self.radius + 1):
                    indices.append((dy, dx))
            return tuple(indices)
        else:
            # von Neumann neighborhood: Manhattan distance <= radius
            indices = []
            for dy in range(-self.radius, self.radius + 1):
                for dx in range(-self.radius, self.radius + 1):
                    if abs(dy) + abs(dx) <= self.radius:
                        indices.append((dy, dx))
            return tuple(indices)
    
    def size_hint(self) -> int:
        """Get total grid size."""
        return self.height * self.width
    
    def __repr__(self) -> str:
        return f"Grid2D(height={self.height}, width={self.width}, radius={self.radius}, type={self.neighborhood_type})"
