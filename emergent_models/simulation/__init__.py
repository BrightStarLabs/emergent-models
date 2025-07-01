"""
Simulation components for cellular automata.

This module provides simulators that perform pure CA evolution
on pre-encoded tapes.
"""

from .simulator import Simulator, BatchSimulator

__all__ = ["Simulator", "BatchSimulator"]
