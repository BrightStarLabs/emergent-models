"""
TapeVisualizer for 1D CA tape state visualization.

This module provides the TapeVisualizer component for creating colorful
visualizations of 1D cellular automaton tape states with annotations
for programme regions, separators, and beacons.
"""

from typing import Optional, Dict, Union, Tuple, List
import numpy as np

from .base import StateAwareVisualizer
from .utils import (
    get_figure_size, format_tape_position, annotate_tape_regions,
    validate_color_scheme
)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
    PlotlyFigure = go.Figure
except ImportError:
    PLOTLY_AVAILABLE = False
    PlotlyFigure = None


class TapeVisualizer(StateAwareVisualizer):
    """
    Visualizer for 1D CA tape states with colorful representation.
    
    This visualizer creates horizontal bar charts showing the state of each
    cell in a CA tape, with customizable colors and annotations for different
    regions (programme, separator, beacons).
    
    Examples
    --------
    >>> from emergent_models.core.state import StateModel
    >>> from emergent_models.visualization import TapeVisualizer
    >>> 
    >>> state_model = StateModel([0, 1, 2, 3])
    >>> tape_viz = TapeVisualizer(state_model)
    >>> 
    >>> # Visualize a tape
    >>> tape = np.array([1, 0, 2, 0, 0, 3, 3, 0, 0, 2])
    >>> fig = tape_viz.render(tape, title="CA Tape State")
    >>> 
    >>> # With annotations for EM-4/3 encoding
    >>> fig = tape_viz.render(tape, prog_len=3, input_val=2, 
    ...                      annotate_regions=True)
    """
    
    def __init__(self, state_model, color_scheme: Optional[Dict] = None, 
                 backend: str = "matplotlib", **kwargs):
        """
        Initialize TapeVisualizer.
        
        Parameters
        ----------
        state_model : StateModel
            CA state model defining valid states
        color_scheme : dict, optional
            Custom color mapping for states
        backend : str, default="matplotlib"
            Visualization backend ("matplotlib" or "plotly")
        **kwargs
            Additional visualizer options
        """
        super().__init__(state_model, color_scheme, backend=backend, **kwargs)
        
        # Visualization settings
        self.cell_height = kwargs.get('cell_height', 0.8)
        self.show_grid = kwargs.get('show_grid', True)
        self.show_labels = kwargs.get('show_labels', True)
    
    def render(self, tape: np.ndarray, title: str = "CA Tape State",
               prog_len: Optional[int] = None, input_val: Optional[int] = None,
               annotate_regions: bool = False, zoom_range: Optional[Tuple[int, int]] = None,
               show_positions: bool = True, **kwargs):
        """
        Render 1D tape visualization.
        
        Parameters
        ----------
        tape : np.ndarray
            1D array of tape states
        title : str, default="CA Tape State"
            Plot title
        prog_len : int, optional
            Length of programme region for annotations
        input_val : int, optional
            Input value for beacon position calculation
        annotate_regions : bool, default=False
            Whether to annotate programme/separator/beacon regions
        zoom_range : tuple, optional
            (start, end) positions to zoom into
        show_positions : bool, default=True
            Whether to show position labels
        **kwargs
            Additional plotting options
            
        Returns
        -------
        Figure
            Matplotlib or Plotly figure
        """
        tape = self._validate_tape(tape)
        
        # Apply zoom if specified
        if zoom_range is not None:
            start, end = zoom_range
            start = max(0, start)
            end = min(len(tape), end)
            tape = tape[start:end]
            positions = np.arange(start, end)
        else:
            positions = np.arange(len(tape))
        
        if self.backend == "matplotlib":
            return self._render_matplotlib(tape, positions, title, prog_len, 
                                         input_val, annotate_regions, 
                                         show_positions, **kwargs)
        elif self.backend == "plotly":
            return self._render_plotly(tape, positions, title, prog_len,
                                     input_val, annotate_regions,
                                     show_positions, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _render_matplotlib(self, tape: np.ndarray, positions: np.ndarray,
                          title: str, prog_len: Optional[int], 
                          input_val: Optional[int], annotate_regions: bool,
                          show_positions: bool, **kwargs):
        """Render using matplotlib backend."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for this backend")
        
        # Calculate figure size
        figsize = kwargs.get('figsize', get_figure_size(len(tape)))
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get colors for each cell
        colors = self._get_state_colors(tape)
        
        # Create horizontal bars for each cell
        bars = ax.barh(0, 1, left=positions, height=self.cell_height, 
                      color=colors, edgecolor='black', linewidth=0.5)
        
        # Customize appearance
        ax.set_xlim(-0.5, len(positions) - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Position')
        ax.set_title(title)
        
        # Remove y-axis
        ax.set_yticks([])
        
        # Position labels
        if show_positions and len(tape) <= 50:  # Only show for reasonable sizes
            ax.set_xticks(positions[::max(1, len(positions)//20)])
            if prog_len is not None:
                # Use semantic labels
                labels = [format_tape_position(pos, prog_len, input_val) 
                         for pos in positions[::max(1, len(positions)//20)]]
                ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add state value annotations on cells
        if len(tape) <= 30:  # Only for small tapes
            for i, (pos, state) in enumerate(zip(positions, tape)):
                ax.text(pos, 0, str(state), ha='center', va='center',
                       color='white' if self._is_dark_color(colors[i]) else 'black',
                       fontweight='bold', fontsize=10)
        
        # Add region annotations
        if annotate_regions and prog_len is not None:
            annotate_tape_regions(ax, tape, prog_len, input_val)
            ax.legend(loc='upper right')
        
        # Add grid
        if self.show_grid:
            ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        return self._store_result(fig, tape)
    
    def _render_plotly(self, tape: np.ndarray, positions: np.ndarray,
                      title: str, prog_len: Optional[int],
                      input_val: Optional[int], annotate_regions: bool,
                      show_positions: bool, **kwargs):
        """Render using plotly backend."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required for this backend")
        
        # Get colors for each cell
        colors = self._get_state_colors(tape)
        
        # Create hover text
        hover_text = []
        for i, (pos, state) in enumerate(zip(positions, tape)):
            if prog_len is not None:
                pos_label = format_tape_position(pos, prog_len, input_val)
            else:
                pos_label = f"Pos[{pos}]"
            hover_text.append(f"{pos_label}<br>State: {state}")
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=positions,
            y=[1] * len(tape),
            marker_color=colors,
            marker_line_color='black',
            marker_line_width=0.5,
            hovertext=hover_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Position',
            yaxis=dict(visible=False),
            height=200,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Add region annotations as shapes
        if annotate_regions and prog_len is not None:
            self._add_plotly_regions(fig, positions, prog_len, input_val)
        
        return self._store_result(fig, tape)
    
    def _add_plotly_regions(self, fig, positions: np.ndarray,
                           prog_len: int, input_val: Optional[int]) -> None:
        """Add region annotations to plotly figure."""
        max_pos = positions[-1] if len(positions) > 0 else 0
        
        # Programme region
        if prog_len > 0:
            fig.add_shape(
                type="rect",
                x0=-0.5, x1=prog_len - 0.5,
                y0=-0.1, y1=1.1,
                fillcolor="green",
                opacity=0.1,
                line=dict(width=0)
            )
        
        # Separator region
        if prog_len < max_pos:
            sep_end = min(prog_len + 2, max_pos + 1)
            fig.add_shape(
                type="rect",
                x0=prog_len - 0.5, x1=sep_end - 0.5,
                y0=-0.1, y1=1.1,
                fillcolor="orange",
                opacity=0.1,
                line=dict(width=0)
            )
        
        # Beacon region
        if input_val is not None:
            beacon_pos = prog_len + 2 + input_val + 1
            if beacon_pos <= max_pos:
                fig.add_shape(
                    type="rect",
                    x0=beacon_pos - 0.5, x1=beacon_pos + 0.5,
                    y0=-0.1, y1=1.1,
                    fillcolor="red",
                    opacity=0.2,
                    line=dict(width=0)
                )
    
    def _is_dark_color(self, color: str) -> bool:
        """Check if color is dark for text contrast."""
        # Simple heuristic for hex colors
        if color.startswith('#') and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            return luminance < 0.5
        return False
