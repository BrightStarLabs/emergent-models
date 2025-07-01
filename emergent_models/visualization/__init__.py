"""
Visualization components for emergent models.

This module provides a comprehensive visualization system following the component-first
design pattern. All visualizers implement the base Visualizer interface and can be
composed together for complex dashboards.

Examples
--------
>>> from emergent_models.visualization import TapeVisualizer, ProgressionVisualizer
>>> 
>>> # Visualize a single tape state
>>> tape_viz = TapeVisualizer(state_model)
>>> fig = tape_viz.render(final_tape)
>>> 
>>> # Create 2D space-time evolution map
>>> prog_viz = ProgressionVisualizer(state_model)
>>> fig = prog_viz.render(simulation_history)
>>> 
>>> # Combine multiple visualizations
>>> from emergent_models.visualization import CombinedVisualizer
>>> dashboard = CombinedVisualizer(tape_viz, prog_viz)
>>> fig = dashboard.render({'tape': final_tape, 'history': simulation_history})
"""

from .base import Visualizer
from .themes import ColorScheme, EM43_COLORS, SCIENTIFIC_COLORS
from .utils import save_figure, create_subplot_grid

# Import visualization components as they become available
try:
    from .tape import TapeVisualizer
except ImportError:
    TapeVisualizer = None

try:
    from .progression import ProgressionVisualizer
except ImportError:
    ProgressionVisualizer = None

try:
    from .fitness import FitnessVisualizer
except ImportError:
    FitnessVisualizer = None

try:
    from .population import PopulationVisualizer
except ImportError:
    PopulationVisualizer = None

try:
    from .interactive import InteractiveVisualizer
except ImportError:
    InteractiveVisualizer = None

try:
    from .combined import CombinedVisualizer
except ImportError:
    CombinedVisualizer = None

try:
    from .live import LiveTrainingVisualizer
except ImportError:
    LiveTrainingVisualizer = None

__all__ = [
    # Base classes
    'Visualizer',
    'ColorScheme',
    
    # Color schemes
    'EM43_COLORS',
    'SCIENTIFIC_COLORS',
    
    # Utilities
    'save_figure',
    'create_subplot_grid',
    
    # Visualization components (available when implemented)
    'TapeVisualizer',
    'ProgressionVisualizer', 
    'FitnessVisualizer',
    'PopulationVisualizer',
    'InteractiveVisualizer',
    'CombinedVisualizer',
    'LiveTrainingVisualizer',
]

# Version info
__version__ = "0.1.0"
