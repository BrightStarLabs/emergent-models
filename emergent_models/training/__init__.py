# Legacy trainer import - commented out since file doesn't exist
# from .trainer import CATrainer, create_accuracy_validator
from .checkpointing import save_genome, load_genome

# New architecture components
from .fitness import AbsoluteDifferenceFitness, SparsityPenalizedFitness, ComplexityRewardFitness
from .optimizer import GAOptimizer, RandomSearchOptimizer
from .trainer import Trainer
from .monitor import DetailedMonitor, ProgressMonitor, SilentMonitor, TqdmMonitor, JupyterMonitor, CombinedMonitor

__all__ = [
    # Legacy components (commented out)
    # "CATrainer", "create_accuracy_validator",
    "save_genome", "load_genome",
    # New architecture components
    "AbsoluteDifferenceFitness", "SparsityPenalizedFitness", "ComplexityRewardFitness",
    "GAOptimizer", "RandomSearchOptimizer",
    "Trainer",
    "DetailedMonitor", "ProgressMonitor", "SilentMonitor", "TqdmMonitor", "JupyterMonitor", "CombinedMonitor"
]
