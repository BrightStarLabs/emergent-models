"""
Color schemes and visual themes for CA visualization.

This module provides predefined color schemes and styling options
for consistent and beautiful visualizations across the SDK.
"""

from typing import Dict, List, Optional
import numpy as np

# EM-4/3 specific color scheme
EM43_COLORS = {
    0: '#000000',  # Empty - Black
    1: '#FFFFFF',  # White - White  
    2: '#FF0000',  # Red - Red
    3: '#0000FF',  # Blue - Blue
}

# Scientific publication color scheme (colorblind-friendly)
SCIENTIFIC_COLORS = {
    0: '#2C2C2C',  # Empty - Dark gray
    1: '#F5F5F5',  # White - Light gray
    2: '#E74C3C',  # Red - Bright red (colorblind safe)
    3: '#3498DB',  # Blue - Bright blue (colorblind safe)
}

# High contrast color scheme for accessibility
HIGH_CONTRAST_COLORS = {
    0: '#000000',  # Empty - Black
    1: '#FFFFFF',  # White - White
    2: '#FF6B6B',  # Red - Bright coral
    3: '#4ECDC4',  # Blue - Bright teal
}

# Pastel color scheme for presentations
PASTEL_COLORS = {
    0: '#F8F9FA',  # Empty - Very light gray
    1: '#E9ECEF',  # White - Light gray
    2: '#FFB3BA',  # Red - Pastel pink
    3: '#BAE1FF',  # Blue - Pastel blue
}

# Viridis-inspired color scheme
VIRIDIS_COLORS = {
    0: '#440154',  # Empty - Dark purple
    1: '#31688E',  # White - Blue
    2: '#35B779',  # Red - Green
    3: '#FDE725',  # Blue - Yellow
}


class ColorScheme:
    """
    Flexible color scheme manager for CA visualizations.
    
    Examples
    --------
    >>> scheme = ColorScheme(EM43_COLORS)
    >>> colors = scheme.get_colors([0, 1, 2, 3])
    >>> 
    >>> # Create custom scheme
    >>> custom = ColorScheme({0: 'black', 1: 'white', 2: 'red', 3: 'blue'})
    >>> 
    >>> # Use with matplotlib colormap
    >>> cmap = scheme.to_matplotlib_cmap()
    """
    
    def __init__(self, color_mapping: Dict[int, str], name: str = "custom"):
        """
        Initialize color scheme.
        
        Parameters
        ----------
        color_mapping : dict
            Mapping from state values to color strings
        name : str, default="custom"
            Name of the color scheme
        """
        self.mapping = color_mapping
        self.name = name
        self.states = sorted(color_mapping.keys())
    
    def get_color(self, state: int, default: str = '#808080') -> str:
        """Get color for a specific state."""
        return self.mapping.get(state, default)
    
    def get_colors(self, states: List[int]) -> List[str]:
        """Get list of colors for multiple states."""
        return [self.get_color(state) for state in states]
    
    def to_matplotlib_cmap(self):
        """Convert to matplotlib colormap."""
        try:
            from matplotlib.colors import ListedColormap
            colors = [self.mapping[state] for state in self.states]
            return ListedColormap(colors, name=self.name)
        except ImportError:
            raise ImportError("matplotlib is required for colormap conversion")
    
    def to_plotly_colorscale(self) -> List[List]:
        """Convert to plotly colorscale format."""
        n_states = len(self.states)
        colorscale = []
        
        for i, state in enumerate(self.states):
            position = i / (n_states - 1) if n_states > 1 else 0
            colorscale.append([position, self.mapping[state]])
        
        return colorscale
    
    def preview(self, figsize: tuple = (8, 2)):
        """Create a preview of the color scheme."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(figsize=figsize)
            
            for i, (state, color) in enumerate(self.mapping.items()):
                rect = patches.Rectangle((i, 0), 1, 1, facecolor=color, 
                                       edgecolor='black', linewidth=1)
                ax.add_patch(rect)
                ax.text(i + 0.5, 0.5, str(state), ha='center', va='center',
                       color='white' if self._is_dark_color(color) else 'black',
                       fontweight='bold')
            
            ax.set_xlim(0, len(self.mapping))
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.set_title(f'Color Scheme: {self.name}')
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            print(f"Color scheme '{self.name}':")
            for state, color in self.mapping.items():
                print(f"  State {state}: {color}")
    
    def _is_dark_color(self, color: str) -> bool:
        """Check if a color is dark (for text contrast)."""
        # Simple heuristic based on hex color
        if color.startswith('#') and len(color) == 7:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16) 
            b = int(color[5:7], 16)
            # Calculate luminance
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            return luminance < 0.5
        return False


# Predefined color scheme objects
EM43_SCHEME = ColorScheme(EM43_COLORS, "EM-4/3")
SCIENTIFIC_SCHEME = ColorScheme(SCIENTIFIC_COLORS, "Scientific")
HIGH_CONTRAST_SCHEME = ColorScheme(HIGH_CONTRAST_COLORS, "High Contrast")
PASTEL_SCHEME = ColorScheme(PASTEL_COLORS, "Pastel")
VIRIDIS_SCHEME = ColorScheme(VIRIDIS_COLORS, "Viridis")

# Default scheme
DEFAULT_SCHEME = EM43_SCHEME


def get_scheme(name: str) -> ColorScheme:
    """
    Get predefined color scheme by name.
    
    Parameters
    ----------
    name : str
        Name of the color scheme
        
    Returns
    -------
    ColorScheme
        The requested color scheme
        
    Raises
    ------
    ValueError
        If scheme name is not recognized
    """
    schemes = {
        'em43': EM43_SCHEME,
        'scientific': SCIENTIFIC_SCHEME,
        'high_contrast': HIGH_CONTRAST_SCHEME,
        'pastel': PASTEL_SCHEME,
        'viridis': VIRIDIS_SCHEME,
        'default': DEFAULT_SCHEME,
    }
    
    name_lower = name.lower()
    if name_lower not in schemes:
        available = ', '.join(schemes.keys())
        raise ValueError(f"Unknown color scheme '{name}'. Available: {available}")
    
    return schemes[name_lower]


def list_schemes() -> List[str]:
    """List all available color scheme names."""
    return ['em43', 'scientific', 'high_contrast', 'pastel', 'viridis', 'default']
