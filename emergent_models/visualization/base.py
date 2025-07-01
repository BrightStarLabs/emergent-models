"""
Base visualization classes and interfaces.

This module defines the core Visualizer interface that all visualization components
must implement, following the component-first design pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Union
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.figure
    MATPLOTLIB_AVAILABLE = True
    Figure = matplotlib.figure.Figure
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = Any

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
    PlotlyFigure = go.Figure
except ImportError:
    PLOTLY_AVAILABLE = False
    PlotlyFigure = Any


class Visualizer(ABC):
    """
    Abstract base class for all visualization components.
    
    This class defines the interface that all visualizers must implement,
    ensuring consistency across the visualization system.
    
    Examples
    --------
    >>> class MyVisualizer(Visualizer):
    ...     def render(self, data, **kwargs):
    ...         fig, ax = plt.subplots()
    ...         ax.plot(data)
    ...         return fig
    ...     
    ...     def save(self, filepath, **kwargs):
    ...         fig = self.render(self.last_data, **kwargs)
    ...         fig.savefig(filepath)
    """
    
    def __init__(self, backend: str = "matplotlib", **kwargs):
        """
        Initialize visualizer with specified backend.
        
        Parameters
        ----------
        backend : str, default="matplotlib"
            Visualization backend to use ("matplotlib", "plotly")
        **kwargs
            Additional backend-specific configuration
        """
        self.backend = backend
        self.config = kwargs
        self.last_data = None
        self.last_figure = None
        
        # Validate backend availability
        if backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required but not installed")
        elif backend == "plotly" and not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required but not installed")
    
    @abstractmethod
    def render(self, data: Any, **kwargs):
        """
        Render visualization from data.
        
        Parameters
        ----------
        data : Any
            Data to visualize (format depends on specific visualizer)
        **kwargs
            Visualization options (title, colors, size, etc.)
            
        Returns
        -------
        Figure
            Matplotlib or Plotly figure object
        """
        pass
    
    def save(self, filepath: str, data: Optional[Any] = None, **kwargs) -> None:
        """
        Save visualization to file.
        
        Parameters
        ----------
        filepath : str
            Output file path
        data : Any, optional
            Data to visualize. If None, uses last rendered data
        **kwargs
            Save options (dpi, format, etc.)
        """
        if data is None:
            if self.last_figure is None:
                raise ValueError("No data provided and no previous figure to save")
            fig = self.last_figure
        else:
            fig = self.render(data, **kwargs)
        
        if self.backend == "matplotlib":
            fig.savefig(filepath, **kwargs)
        elif self.backend == "plotly":
            # Determine format from file extension
            if filepath.endswith('.html'):
                fig.write_html(filepath)
            elif filepath.endswith('.png'):
                fig.write_image(filepath)
            elif filepath.endswith('.svg'):
                fig.write_image(filepath)
            else:
                fig.write_html(filepath + '.html')
    
    def show(self, data: Optional[Any] = None, **kwargs):
        """
        Render and display visualization.
        
        Parameters
        ----------
        data : Any, optional
            Data to visualize. If None, uses last rendered data
        **kwargs
            Visualization options
            
        Returns
        -------
        Figure
            The rendered figure
        """
        if data is None:
            if self.last_figure is None:
                raise ValueError("No data provided and no previous figure to show")
            fig = self.last_figure
        else:
            fig = self.render(data, **kwargs)
        
        if self.backend == "matplotlib":
            plt.show()
        elif self.backend == "plotly":
            fig.show()
        
        return fig
    
    def _store_result(self, fig, data: Any):
        """Store figure and data for later use."""
        self.last_figure = fig
        self.last_data = data
        return fig


class StateAwareVisualizer(Visualizer):
    """
    Base class for visualizers that work with CA state models.
    
    This class provides common functionality for visualizers that need
    to understand CA states and their visual representation.
    """
    
    def __init__(self, state_model, color_scheme: Optional[Dict] = None, **kwargs):
        """
        Initialize state-aware visualizer.
        
        Parameters
        ----------
        state_model : StateModel
            CA state model defining valid states
        color_scheme : dict, optional
            Mapping from state values to colors
        **kwargs
            Additional visualizer options
        """
        super().__init__(**kwargs)
        self.state_model = state_model
        self.color_scheme = color_scheme or self._default_color_scheme()
    
    def _default_color_scheme(self) -> Dict[int, str]:
        """Create default color scheme for states."""
        from .themes import EM43_COLORS
        return EM43_COLORS
    
    def _validate_tape(self, tape: np.ndarray) -> np.ndarray:
        """Validate that tape contains only valid states."""
        tape = np.asarray(tape)
        valid_states = set(self.state_model.symbols)
        tape_states = set(np.unique(tape))

        invalid_states = tape_states - valid_states
        if invalid_states:
            raise ValueError(f"Tape contains invalid states: {invalid_states}")

        return tape
    
    def _get_state_colors(self, tape: np.ndarray) -> list:
        """Get list of colors corresponding to tape states."""
        return [self.color_scheme.get(state, '#808080') for state in tape]
