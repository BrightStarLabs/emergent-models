"""
FitnessVisualizer for training progress and convergence analysis.

This module provides the FitnessVisualizer component for creating
comprehensive visualizations of genetic algorithm training progress,
including fitness evolution, convergence analysis, and performance metrics.
"""

from typing import Optional, Dict, List
import numpy as np

from .base import Visualizer
from .utils import create_subplot_grid

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class FitnessVisualizer(Visualizer):
    """
    Visualizer for training progress and fitness evolution analysis.
    
    Creates comprehensive dashboards showing:
    - Best/mean/std fitness over generations
    - Convergence analysis and trends
    - Performance metrics (time per generation)
    - Population diversity indicators
    
    Examples
    --------
    >>> from emergent_models.visualization import FitnessVisualizer
    >>> 
    >>> # Create visualizer
    >>> fitness_viz = FitnessVisualizer()
    >>> 
    >>> # Visualize training results
    >>> training_history = {
    ...     'best_fitness': [best_scores_per_gen],
    ...     'mean_fitness': [mean_scores_per_gen],
    ...     'std_fitness': [std_scores_per_gen],
    ...     'generation_times': [times_per_gen]
    ... }
    >>> fig = fitness_viz.render(training_history, title="Training Progress")
    >>> 
    >>> # Focus on convergence analysis
    >>> fig = fitness_viz.render(training_history, show_convergence=True)
    """
    
    def __init__(self, backend: str = "matplotlib", **kwargs):
        """
        Initialize FitnessVisualizer.
        
        Parameters
        ----------
        backend : str, default="matplotlib"
            Visualization backend ("matplotlib" or "plotly")
        **kwargs
            Additional visualizer options
        """
        super().__init__(backend=backend, **kwargs)
        
        # Visualization settings
        self.show_std_bands = kwargs.get('show_std_bands', True)
        self.show_performance = kwargs.get('show_performance', True)
        self.smooth_curves = kwargs.get('smooth_curves', False)
        self.smoothing_window = kwargs.get('smoothing_window', 5)
    
    def render(self, training_history: Dict, title: str = "Training Progress",
               show_convergence: bool = True, show_performance: bool = None,
               highlight_best: bool = True, **kwargs):
        """
        Render training progress visualization.
        
        Parameters
        ----------
        training_history : dict
            Training history with keys: 'best_fitness', 'mean_fitness', 
            'std_fitness', 'generation_times'
        title : str, default="Training Progress"
            Main plot title
        show_convergence : bool, default=True
            Whether to include convergence analysis
        show_performance : bool, optional
            Whether to show performance metrics (default: auto-detect)
        highlight_best : bool, default=True
            Whether to highlight best fitness achieved
        **kwargs
            Additional plotting options
            
        Returns
        -------
        Figure
            Matplotlib or Plotly figure
        """
        # Validate input data
        required_keys = ['best_fitness', 'mean_fitness', 'std_fitness']
        for key in required_keys:
            if key not in training_history:
                raise ValueError(f"Missing required key in training_history: {key}")
        
        # Auto-detect performance data
        if show_performance is None:
            show_performance = 'generation_times' in training_history
        
        if self.backend == "matplotlib":
            return self._render_matplotlib(training_history, title, show_convergence,
                                         show_performance, highlight_best, **kwargs)
        elif self.backend == "plotly":
            return self._render_plotly(training_history, title, show_convergence,
                                     show_performance, highlight_best, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _render_matplotlib(self, history: Dict, title: str, show_convergence: bool,
                          show_performance: bool, highlight_best: bool, **kwargs):
        """Render using matplotlib backend."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for this backend")
        
        # Determine subplot layout
        n_plots = 1 + int(show_convergence) + int(show_performance)
        
        if n_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=kwargs.get('figsize', (10, 6)))
            axes = [axes]
        elif n_plots == 2:
            fig, axes = plt.subplots(2, 1, figsize=kwargs.get('figsize', (10, 10)))
        else:  # n_plots == 3
            fig, axes = plt.subplots(2, 2, figsize=kwargs.get('figsize', (15, 10)))
            axes = axes.flatten()
            # Hide the unused subplot
            axes[3].set_visible(False)
        
        # Extract data
        generations = np.arange(len(history['best_fitness']))
        best_fitness = np.array(history['best_fitness'])
        mean_fitness = np.array(history['mean_fitness'])
        std_fitness = np.array(history['std_fitness'])
        
        # Apply smoothing if requested
        if self.smooth_curves:
            best_fitness = self._smooth_curve(best_fitness)
            mean_fitness = self._smooth_curve(mean_fitness)
            std_fitness = self._smooth_curve(std_fitness)
        
        # Plot 1: Fitness Evolution
        ax = axes[0]
        ax.plot(generations, best_fitness, 'g-', linewidth=2, label='Best Fitness')
        ax.plot(generations, mean_fitness, 'b-', linewidth=2, label='Mean Fitness')
        
        # Add standard deviation bands
        if self.show_std_bands:
            ax.fill_between(generations, 
                           mean_fitness - std_fitness,
                           mean_fitness + std_fitness,
                           alpha=0.3, color='blue', label='±1 Std Dev')
        
        # Highlight best fitness achieved
        if highlight_best:
            best_idx = np.argmax(best_fitness)
            ax.scatter(generations[best_idx], best_fitness[best_idx], 
                      color='red', s=100, zorder=5, label=f'Best: {best_fitness[best_idx]:.4f}')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_idx = 1
        
        # Plot 2: Convergence Analysis
        if show_convergence:
            ax = axes[plot_idx]
            
            # Calculate improvement rate
            improvement = np.diff(best_fitness)

            ax.plot(generations[1:], improvement, 'r-', alpha=0.5, label='Raw Improvement')

            # Add smoothed improvement if we have enough data
            if len(improvement) > self.smoothing_window:
                improvement_smooth = self._smooth_curve(improvement)
                # Adjust x-axis for smoothed data (moving average reduces length)
                smooth_generations = generations[self.smoothing_window:]
                ax.plot(smooth_generations, improvement_smooth, 'r-', linewidth=2, label='Smoothed Improvement')
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness Improvement')
            ax.set_title('Convergence Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Plot 3: Performance Metrics
        if show_performance and 'generation_times' in history:
            ax = axes[plot_idx]
            
            gen_times = np.array(history['generation_times'])
            ax.plot(generations, gen_times, 'purple', linewidth=2, label='Generation Time')
            
            # Add moving average
            if len(gen_times) > 5:
                moving_avg = self._moving_average(gen_times, window=5)
                ax.plot(generations[4:], moving_avg, 'orange', linewidth=2, 
                       linestyle='--', label='5-Gen Moving Avg')
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Performance Metrics')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return self._store_result(fig, history)
    
    def _render_plotly(self, history: Dict, title: str, show_convergence: bool,
                      show_performance: bool, highlight_best: bool, **kwargs):
        """Render using plotly backend."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required for this backend")
        
        # Determine subplot layout
        n_plots = 1 + int(show_convergence) + int(show_performance)
        
        if n_plots == 1:
            fig = go.Figure()
            subplot_titles = None
        else:
            subplot_titles = ['Fitness Evolution']
            if show_convergence:
                subplot_titles.append('Convergence Analysis')
            if show_performance:
                subplot_titles.append('Performance Metrics')
            
            if n_plots == 2:
                fig = make_subplots(rows=2, cols=1, subplot_titles=subplot_titles)
            else:  # n_plots == 3
                fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles)
        
        # Extract data
        generations = np.arange(len(history['best_fitness']))
        best_fitness = np.array(history['best_fitness'])
        mean_fitness = np.array(history['mean_fitness'])
        std_fitness = np.array(history['std_fitness'])
        
        # Apply smoothing if requested
        if self.smooth_curves:
            best_fitness = self._smooth_curve(best_fitness)
            mean_fitness = self._smooth_curve(mean_fitness)
            std_fitness = self._smooth_curve(std_fitness)
        
        # Plot 1: Fitness Evolution
        row, col = (1, 1) if n_plots > 1 else (None, None)
        
        # Best fitness line
        fig.add_trace(go.Scatter(
            x=generations, y=best_fitness,
            mode='lines', name='Best Fitness',
            line=dict(color='green', width=3)
        ), row=row, col=col)
        
        # Mean fitness line
        fig.add_trace(go.Scatter(
            x=generations, y=mean_fitness,
            mode='lines', name='Mean Fitness',
            line=dict(color='blue', width=3)
        ), row=row, col=col)
        
        # Standard deviation bands
        if self.show_std_bands:
            fig.add_trace(go.Scatter(
                x=np.concatenate([generations, generations[::-1]]),
                y=np.concatenate([mean_fitness + std_fitness, 
                                (mean_fitness - std_fitness)[::-1]]),
                fill='toself', fillcolor='rgba(0,0,255,0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev', showlegend=True
            ), row=row, col=col)
        
        # Highlight best fitness
        if highlight_best:
            best_idx = np.argmax(best_fitness)
            fig.add_trace(go.Scatter(
                x=[generations[best_idx]], y=[best_fitness[best_idx]],
                mode='markers', name=f'Best: {best_fitness[best_idx]:.4f}',
                marker=dict(color='red', size=10)
            ), row=row, col=col)
        
        # Plot 2: Convergence Analysis
        if show_convergence:
            improvement = np.diff(best_fitness)
            
            fig.add_trace(go.Scatter(
                x=generations[1:], y=improvement,
                mode='lines', name='Fitness Improvement',
                line=dict(color='red', width=2)
            ), row=2, col=1)
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", 
                         opacity=0.5, row=2, col=1)
        
        # Plot 3: Performance Metrics
        if show_performance and 'generation_times' in history:
            gen_times = np.array(history['generation_times'])
            
            row_idx = 2 if show_convergence else 1
            col_idx = 2 if show_convergence else 1
            
            fig.add_trace(go.Scatter(
                x=generations, y=gen_times,
                mode='lines', name='Generation Time',
                line=dict(color='purple', width=2)
            ), row=row_idx, col=col_idx)
        
        # Update layout
        fig.update_layout(
            title=title,
            height=600 if n_plots == 1 else 800,
            showlegend=True
        )
        
        return self._store_result(fig, history)
    
    def _smooth_curve(self, data: np.ndarray) -> np.ndarray:
        """Apply smoothing to curve data."""
        if len(data) < self.smoothing_window:
            return data
        
        # Simple moving average smoothing
        return self._moving_average(data, self.smoothing_window)
    
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average."""
        if len(data) < window:
            return data
        
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def analyze_convergence(self, training_history: Dict) -> Dict:
        """
        Analyze convergence characteristics of training.
        
        Parameters
        ----------
        training_history : dict
            Training history data
            
        Returns
        -------
        dict
            Convergence analysis results
        """
        best_fitness = np.array(training_history['best_fitness'])
        
        # Calculate improvement metrics
        total_improvement = best_fitness[-1] - best_fitness[0]
        improvement_rate = np.diff(best_fitness)
        
        # Find convergence point (when improvement becomes negligible)
        threshold = 0.001 * abs(total_improvement)
        recent_improvements = improvement_rate[-10:]  # Last 10 generations
        converged = np.all(np.abs(recent_improvements) < threshold)
        
        # Calculate convergence generation
        convergence_gen = None
        if converged:
            for i in range(len(improvement_rate) - 10, -1, -1):
                if abs(improvement_rate[i]) >= threshold:
                    convergence_gen = i + 1
                    break
        
        return {
            'total_improvement': total_improvement,
            'final_fitness': best_fitness[-1],
            'converged': converged,
            'convergence_generation': convergence_gen,
            'average_improvement_rate': np.mean(improvement_rate),
            'improvement_std': np.std(improvement_rate)
        }
