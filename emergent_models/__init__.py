"""
Emergent Models - Component-First Cellular Automata Library
"""

from .__version__ import __version__

# New Component-First Architecture
# Core components
from .core.state import StateModel
from .core.space_model import SpaceModel, Tape1D, Grid2D
from .core.base import Encoder, FitnessFn, Monitor, ConsoleLogger

# Rules and genome
from .rules.ruleset import RuleSet, create_em43_ruleset, create_elementary_ruleset
from .rules.programme import Programme, create_random_programme
from .rules.sanitization import lut_idx
from .genome import Genome

# Encoders
from .encoders.em43 import Em43Encoder
from .encoders.new_binary import BinaryEncoder

# Simulation
from .simulation.simulator import Simulator, BatchSimulator

# Training
from .training.fitness import (
    AbsoluteDifferenceFitness,
    SparsityPenalizedFitness, ComplexityRewardFitness
)
from .training.optimizer import GAOptimizer, RandomSearchOptimizer
from .training.trainer import Trainer
from .training.monitor import (
    DetailedMonitor, ProgressMonitor, SilentMonitor,
    TqdmMonitor, JupyterMonitor, CombinedMonitor
)
from .training.checkpointing import save_genome, load_genome

__all__ = [
    # Version
    '__version__',

    # Core components
    'StateModel', 'SpaceModel', 'Tape1D', 'Grid2D',
    'Encoder', 'FitnessFn', 'Monitor', 'ConsoleLogger',

    # Rules and genome
    'RuleSet', 'create_em43_ruleset', 'create_elementary_ruleset',
    'Programme', 'create_random_programme', 'lut_idx',
    'Genome',

    # Encoders
    'Em43Encoder',

    # Simulation
    'Simulator', 'BatchSimulator',

    # Training
    'AbsoluteDifferenceFitness',
    'SparsityPenalizedFitness', 'ComplexityRewardFitness',
    'GAOptimizer', 'RandomSearchOptimizer',
    'Trainer',
    'DetailedMonitor', 'ProgressMonitor', 'SilentMonitor',
    'TqdmMonitor', 'JupyterMonitor', 'CombinedMonitor',
    'save_genome', 'load_genome'
]

