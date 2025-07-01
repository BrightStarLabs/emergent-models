"""
Utility functions for visualization components.

This module provides common helper functions used across different
visualization components.
"""

from typing import Tuple, List, Optional, Union, Any
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
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    PlotlyFigure = go.Figure
except ImportError:
    PLOTLY_AVAILABLE = False
    PlotlyFigure = Any


def save_figure(fig: Union[Figure, PlotlyFigure], filepath: str, 
                backend: str = "matplotlib", **kwargs) -> None:
    """
    Save figure to file with appropriate format detection.
    
    Parameters
    ----------
    fig : Figure or PlotlyFigure
        Figure to save
    filepath : str
        Output file path
    backend : str, default="matplotlib"
        Backend used to create the figure
    **kwargs
        Additional save options
    """
    if backend == "matplotlib":
        # Default matplotlib save options
        save_kwargs = {
            'dpi': 300,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        save_kwargs.update(kwargs)
        fig.savefig(filepath, **save_kwargs)
        
    elif backend == "plotly":
        # Determine format from file extension
        if filepath.endswith('.html'):
            fig.write_html(filepath, **kwargs)
        elif filepath.endswith('.png'):
            save_kwargs = {'width': 1200, 'height': 800, 'scale': 2}
            save_kwargs.update(kwargs)
            fig.write_image(filepath, **save_kwargs)
        elif filepath.endswith('.svg'):
            save_kwargs = {'width': 1200, 'height': 800}
            save_kwargs.update(kwargs)
            fig.write_image(filepath, **save_kwargs)
        elif filepath.endswith('.pdf'):
            save_kwargs = {'width': 1200, 'height': 800}
            save_kwargs.update(kwargs)
            fig.write_image(filepath, **save_kwargs)
        else:
            # Default to HTML
            fig.write_html(filepath + '.html', **kwargs)


def create_subplot_grid(n_plots: int, ncols: int = 2, 
                       backend: str = "matplotlib", **kwargs) -> Tuple[Any, Any]:
    """
    Create subplot grid for multiple visualizations.
    
    Parameters
    ----------
    n_plots : int
        Number of subplots needed
    ncols : int, default=2
        Number of columns in the grid
    backend : str, default="matplotlib"
        Visualization backend
    **kwargs
        Additional subplot options
        
    Returns
    -------
    fig, axes
        Figure and axes objects (format depends on backend)
    """
    nrows = (n_plots + ncols - 1) // ncols  # Ceiling division
    
    if backend == "matplotlib":
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required")
        
        figsize = kwargs.pop('figsize', (6 * ncols, 4 * nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        
        # Ensure axes is always 2D array
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)
        
        # Hide unused subplots
        for i in range(n_plots, nrows * ncols):
            row, col = divmod(i, ncols)
            axes[row, col].set_visible(False)
        
        return fig, axes
        
    elif backend == "plotly":
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required")
        
        subplot_titles = kwargs.pop('subplot_titles', None)
        fig = make_subplots(
            rows=nrows, 
            cols=ncols,
            subplot_titles=subplot_titles,
            **kwargs
        )
        
        return fig, None  # Plotly doesn't return separate axes objects
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


def format_tape_position(pos: int, prog_len: int, input_val: Optional[int] = None) -> str:
    """
    Format tape position with semantic meaning for EM-4/3 encoding.
    
    Parameters
    ----------
    pos : int
        Position on the tape
    prog_len : int
        Length of the programme
    input_val : int, optional
        Input value (for beacon position calculation)
        
    Returns
    -------
    str
        Formatted position description
    """
    if pos < prog_len:
        return f"Prog[{pos}]"
    elif pos == prog_len:
        return "Sep[0]"
    elif pos == prog_len + 1:
        return "Sep[1]"
    elif input_val is not None:
        expected_beacon = prog_len + 2 + input_val + 1
        if pos == expected_beacon:
            return f"Beacon({input_val})"
        elif pos < expected_beacon:
            return f"Zero[{pos - prog_len - 2}]"
        else:
            return f"Pos[{pos}]"
    else:
        return f"Pos[{pos}]"


def annotate_tape_regions(ax, tape: np.ndarray, prog_len: int, 
                         input_val: Optional[int] = None) -> None:
    """
    Add region annotations to a tape visualization.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate
    tape : np.ndarray
        Tape array
    prog_len : int
        Length of the programme
    input_val : int, optional
        Input value for beacon annotation
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    tape_len = len(tape)
    
    # Programme region
    if prog_len > 0:
        ax.axvspan(-0.5, prog_len - 0.5, alpha=0.1, color='green', 
                  label='Programme')
    
    # Separator region
    if prog_len < tape_len:
        sep_end = min(prog_len + 2, tape_len)
        ax.axvspan(prog_len - 0.5, sep_end - 0.5, alpha=0.1, color='orange',
                  label='Separator')
    
    # Beacon region (if input provided)
    if input_val is not None:
        beacon_pos = prog_len + 2 + input_val + 1
        if beacon_pos < tape_len:
            ax.axvspan(beacon_pos - 0.5, beacon_pos + 0.5, alpha=0.2, 
                      color='red', label=f'Beacon({input_val})')


def get_figure_size(n_elements: int, element_width: float = 0.5, 
                   min_width: float = 8.0, max_width: float = 20.0) -> Tuple[float, float]:
    """
    Calculate appropriate figure size based on number of elements.
    
    Parameters
    ----------
    n_elements : int
        Number of elements to display
    element_width : float, default=0.5
        Width per element in inches
    min_width : float, default=8.0
        Minimum figure width
    max_width : float, default=20.0
        Maximum figure width
        
    Returns
    -------
    tuple
        (width, height) in inches
    """
    width = max(min_width, min(max_width, n_elements * element_width))
    height = width * 0.3  # Aspect ratio for tape visualizations
    return width, height


def create_animation_frames(history: List[np.ndarray], 
                          interval: int = 200) -> List[dict]:
    """
    Create animation frames for CA evolution.
    
    Parameters
    ----------
    history : list of np.ndarray
        List of tape states over time
    interval : int, default=200
        Interval between frames in milliseconds
        
    Returns
    -------
    list
        List of frame dictionaries for animation
    """
    frames = []
    for i, tape in enumerate(history):
        frame = {
            'data': tape,
            'name': f'Step {i}',
            'interval': interval
        }
        frames.append(frame)
    
    return frames


def validate_color_scheme(color_scheme: dict, states: List[int]) -> dict:
    """
    Validate and complete color scheme for given states.
    
    Parameters
    ----------
    color_scheme : dict
        Mapping from states to colors
    states : list
        List of valid states
        
    Returns
    -------
    dict
        Validated and completed color scheme
    """
    from .themes import DEFAULT_SCHEME
    
    # Start with default scheme
    validated = DEFAULT_SCHEME.mapping.copy()
    
    # Update with provided colors
    validated.update(color_scheme)
    
    # Ensure all states have colors
    default_colors = ['#808080', '#A0A0A0', '#C0C0C0', '#E0E0E0']
    for i, state in enumerate(states):
        if state not in validated:
            color_idx = i % len(default_colors)
            validated[state] = default_colors[color_idx]
    
    return validated
