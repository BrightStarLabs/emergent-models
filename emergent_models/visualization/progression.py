"""
ProgressionVisualizer for 2D space-time CA evolution visualization.

This module provides the ProgressionVisualizer component for creating
2D heatmaps showing CA evolution over time, with X=position, Y=time_step,
and Color=state.
"""

from typing import Optional, Dict, Union, Tuple, List
import numpy as np

from .base import StateAwareVisualizer
from .utils import get_figure_size, format_tape_position

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
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


class ProgressionVisualizer(StateAwareVisualizer):
    """
    Visualizer for 2D space-time CA evolution maps.
    
    Creates heatmaps showing how CA tape states evolve over time, with
    position on X-axis, time steps on Y-axis, and colors representing
    different states.
    
    Examples
    --------
    >>> from emergent_models.visualization import ProgressionVisualizer
    >>> 
    >>> # Create visualizer
    >>> prog_viz = ProgressionVisualizer(state_model)
    >>> 
    >>> # Visualize evolution history
    >>> history = [tape_step1, tape_step2, tape_step3, ...]
    >>> fig = prog_viz.render(history, title="CA Evolution")
    >>> 
    >>> # With animation
    >>> fig = prog_viz.render(history, animate=True, interval=200)
    >>> 
    >>> # Focus on specific region
    >>> fig = prog_viz.render(history, zoom_range=(10, 50))
    """
    
    def __init__(self, state_model, color_scheme: Optional[Dict] = None,
                 backend: str = "matplotlib", **kwargs):
        """
        Initialize ProgressionVisualizer.
        
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
        
        # Animation settings
        self.default_interval = kwargs.get('interval', 200)  # ms between frames
        self.show_colorbar = kwargs.get('show_colorbar', True)
    
    def render(self, evolution_history: List[np.ndarray], 
               title: str = "CA Evolution",
               prog_len: Optional[int] = None, 
               input_val: Optional[int] = None,
               zoom_range: Optional[Tuple[int, int]] = None,
               animate: bool = False,
               interval: Optional[int] = None,
               show_annotations: bool = True,
               **kwargs):
        """
        Render 2D space-time evolution visualization.
        
        Parameters
        ----------
        evolution_history : list of np.ndarray
            List of tape states over time
        title : str, default="CA Evolution"
            Plot title
        prog_len : int, optional
            Programme length for annotations
        input_val : int, optional
            Input value for beacon annotations
        zoom_range : tuple, optional
            (start, end) positions to focus on
        animate : bool, default=False
            Whether to create animated visualization
        interval : int, optional
            Animation interval in milliseconds
        show_annotations : bool, default=True
            Whether to show region annotations
        **kwargs
            Additional plotting options
            
        Returns
        -------
        Figure
            Matplotlib or Plotly figure (with animation if requested)
        """
        if not evolution_history:
            raise ValueError("Evolution history cannot be empty")
        
        # Validate all tapes
        validated_history = [self._validate_tape(tape) for tape in evolution_history]
        
        # Create 2D array: rows=time_steps, cols=positions
        evolution_matrix = self._create_evolution_matrix(validated_history, zoom_range)
        
        if self.backend == "matplotlib":
            return self._render_matplotlib(evolution_matrix, validated_history,
                                         title, prog_len, input_val, 
                                         zoom_range, animate, interval,
                                         show_annotations, **kwargs)
        elif self.backend == "plotly":
            return self._render_plotly(evolution_matrix, validated_history,
                                     title, prog_len, input_val,
                                     zoom_range, animate, interval,
                                     show_annotations, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _create_evolution_matrix(self, history: List[np.ndarray], 
                                zoom_range: Optional[Tuple[int, int]]) -> np.ndarray:
        """Create 2D matrix from evolution history."""
        if zoom_range is not None:
            start, end = zoom_range
            # Apply zoom to all tapes
            zoomed_history = []
            for tape in history:
                start_idx = max(0, start)
                end_idx = min(len(tape), end)
                zoomed_history.append(tape[start_idx:end_idx])
            history = zoomed_history
        
        # Find maximum tape length
        max_length = max(len(tape) for tape in history)
        
        # Create matrix with padding
        matrix = np.zeros((len(history), max_length), dtype=int)
        for i, tape in enumerate(history):
            matrix[i, :len(tape)] = tape
        
        return matrix
    
    def _render_matplotlib(self, evolution_matrix: np.ndarray,
                          history: List[np.ndarray], title: str,
                          prog_len: Optional[int], input_val: Optional[int],
                          zoom_range: Optional[Tuple[int, int]],
                          animate: bool, interval: Optional[int],
                          show_annotations: bool, **kwargs):
        """Render using matplotlib backend."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for this backend")
        
        n_steps, n_positions = evolution_matrix.shape
        
        # Calculate figure size
        aspect_ratio = kwargs.get('aspect_ratio', 0.5)
        figsize = kwargs.get('figsize', (12, 6))
        
        if animate:
            return self._create_matplotlib_animation(history, title, prog_len,
                                                   input_val, zoom_range,
                                                   interval, show_annotations,
                                                   **kwargs)
        
        # Static heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create colormap
        cmap = self.color_scheme.to_matplotlib_cmap() if hasattr(self.color_scheme, 'to_matplotlib_cmap') else 'viridis'
        
        # Plot heatmap
        im = ax.imshow(evolution_matrix, cmap=cmap, aspect='auto',
                      origin='lower', interpolation='nearest')
        
        # Customize axes
        ax.set_xlabel('Position')
        ax.set_ylabel('Time Step')
        ax.set_title(title)
        
        # Position labels
        if zoom_range is not None:
            start_pos = zoom_range[0]
            positions = np.arange(start_pos, start_pos + n_positions)
        else:
            positions = np.arange(n_positions)
        
        # Set tick labels for reasonable number of positions
        if n_positions <= 50:
            tick_positions = positions[::max(1, n_positions//10)]
            ax.set_xticks(np.arange(len(tick_positions)) * max(1, n_positions//10))
            if prog_len is not None and show_annotations:
                labels = [format_tape_position(pos, prog_len, input_val) 
                         for pos in tick_positions]
                ax.set_xticklabels(labels, rotation=45, ha='right')
            else:
                ax.set_xticklabels(tick_positions)
        
        # Add colorbar
        if self.show_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('State')
        
        # Add region annotations
        if show_annotations and prog_len is not None:
            self._add_matplotlib_annotations(ax, n_steps, prog_len, input_val, zoom_range)
        
        plt.tight_layout()
        return self._store_result(fig, evolution_matrix)
    
    def _render_plotly(self, evolution_matrix: np.ndarray,
                      history: List[np.ndarray], title: str,
                      prog_len: Optional[int], input_val: Optional[int],
                      zoom_range: Optional[Tuple[int, int]],
                      animate: bool, interval: Optional[int],
                      show_annotations: bool, **kwargs):
        """Render using plotly backend."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required for this backend")
        
        n_steps, n_positions = evolution_matrix.shape
        
        if animate:
            return self._create_plotly_animation(history, title, prog_len,
                                               input_val, zoom_range,
                                               interval, show_annotations,
                                               **kwargs)
        
        # Position labels
        if zoom_range is not None:
            start_pos = zoom_range[0]
            x_labels = [f"Pos {start_pos + i}" for i in range(n_positions)]
        else:
            x_labels = [f"Pos {i}" for i in range(n_positions)]
        
        y_labels = [f"Step {i}" for i in range(n_steps)]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=evolution_matrix,
            x=x_labels,
            y=y_labels,
            colorscale=self._get_plotly_colorscale(),
            showscale=self.show_colorbar,
            hovertemplate='Position: %{x}<br>Time: %{y}<br>State: %{z}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Position',
            yaxis_title='Time Step',
            height=600
        )
        
        # Add region annotations
        if show_annotations and prog_len is not None:
            self._add_plotly_annotations(fig, n_steps, prog_len, input_val, zoom_range)
        
        return self._store_result(fig, evolution_matrix)
    
    def _create_matplotlib_animation(self, history: List[np.ndarray],
                                   title: str, prog_len: Optional[int],
                                   input_val: Optional[int],
                                   zoom_range: Optional[Tuple[int, int]],
                                   interval: Optional[int],
                                   show_annotations: bool,
                                   **kwargs) -> plt.Figure:
        """Create matplotlib animation."""
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 3)))
        
        # Initialize with first frame
        first_tape = history[0]
        if zoom_range is not None:
            start, end = zoom_range
            first_tape = first_tape[start:end]
        
        colors = self._get_state_colors(first_tape)
        bars = ax.bar(range(len(first_tape)), [1]*len(first_tape), 
                     color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} - Step 0")
        ax.set_xlabel('Position')
        
        def animate_frame(frame_num):
            tape = history[frame_num]
            if zoom_range is not None:
                start, end = zoom_range
                tape = tape[start:end]
            
            colors = self._get_state_colors(tape)
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_title(f"{title} - Step {frame_num}")
            return bars
        
        anim_interval = interval or self.default_interval
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(history),
                                     interval=anim_interval, blit=False, repeat=True)
        
        # Store animation in figure for access
        fig._animation = anim
        
        return self._store_result(fig, history)
    
    def _create_plotly_animation(self, history: List[np.ndarray],
                               title: str, prog_len: Optional[int],
                               input_val: Optional[int],
                               zoom_range: Optional[Tuple[int, int]],
                               interval: Optional[int],
                               show_annotations: bool,
                               **kwargs):
        """Create plotly animation."""
        # Create frames for animation
        frames = []
        for i, tape in enumerate(history):
            if zoom_range is not None:
                start, end = zoom_range
                tape = tape[start:end]
            
            colors = self._get_state_colors(tape)
            
            frame = go.Frame(
                data=[go.Bar(
                    x=list(range(len(tape))),
                    y=[1] * len(tape),
                    marker_color=colors,
                    marker_line_color='black',
                    marker_line_width=0.5
                )],
                name=f"Step {i}"
            )
            frames.append(frame)
        
        # Create initial figure
        first_tape = history[0]
        if zoom_range is not None:
            start, end = zoom_range
            first_tape = first_tape[start:end]
        
        colors = self._get_state_colors(first_tape)
        
        fig = go.Figure(
            data=[go.Bar(
                x=list(range(len(first_tape))),
                y=[1] * len(first_tape),
                marker_color=colors,
                marker_line_color='black',
                marker_line_width=0.5
            )],
            frames=frames
        )
        
        # Add animation controls
        anim_interval = interval or self.default_interval
        fig.update_layout(
            title=title,
            xaxis_title='Position',
            yaxis=dict(visible=False),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': anim_interval}}]},
                    {'label': 'Pause', 'method': 'animate',
                     'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                ]
            }]
        )
        
        return self._store_result(fig, history)
    
    def _get_plotly_colorscale(self) -> List[List]:
        """Get plotly colorscale from color scheme."""
        if hasattr(self.color_scheme, 'to_plotly_colorscale'):
            return self.color_scheme.to_plotly_colorscale()
        else:
            # Default colorscale
            return [[0, '#000000'], [0.33, '#FFFFFF'], [0.66, '#FF0000'], [1, '#0000FF']]
    
    def _add_matplotlib_annotations(self, ax, n_steps: int, prog_len: int,
                                  input_val: Optional[int],
                                  zoom_range: Optional[Tuple[int, int]]) -> None:
        """Add region annotations to matplotlib plot."""
        offset = zoom_range[0] if zoom_range else 0
        
        # Programme region
        if prog_len > offset:
            ax.axvline(prog_len - offset - 0.5, color='green', alpha=0.7, 
                      linestyle='--', label='Programme End')
        
        # Separator region
        sep_end = prog_len + 2
        if sep_end > offset:
            ax.axvline(sep_end - offset - 0.5, color='orange', alpha=0.7,
                      linestyle='--', label='Separator End')
        
        # Beacon position
        if input_val is not None:
            beacon_pos = prog_len + 2 + input_val + 1
            if beacon_pos > offset:
                ax.axvline(beacon_pos - offset, color='red', alpha=0.7,
                          linestyle='-', linewidth=2, label=f'Beacon({input_val})')
        
        ax.legend()
    
    def _add_plotly_annotations(self, fig, n_steps: int, prog_len: int,
                              input_val: Optional[int],
                              zoom_range: Optional[Tuple[int, int]]) -> None:
        """Add region annotations to plotly plot."""
        offset = zoom_range[0] if zoom_range else 0
        
        # Programme region
        if prog_len > offset:
            fig.add_vline(x=prog_len - offset - 0.5, line_color='green',
                         line_dash='dash', opacity=0.7)
        
        # Separator region  
        sep_end = prog_len + 2
        if sep_end > offset:
            fig.add_vline(x=sep_end - offset - 0.5, line_color='orange',
                         line_dash='dash', opacity=0.7)
        
        # Beacon position
        if input_val is not None:
            beacon_pos = prog_len + 2 + input_val + 1
            if beacon_pos > offset:
                fig.add_vline(x=beacon_pos - offset, line_color='red',
                             line_width=3, opacity=0.7)
