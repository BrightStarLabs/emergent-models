"""
Advanced monitoring and telemetry for CA training.

This module provides detailed monitoring capabilities including
complex telemetry and progress tracking.
"""

import numpy as np
import time
from typing import Dict, Any, Optional
import json
import os

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True

    # Try to import notebook-specific tqdm for better Jupyter support
    try:
        from tqdm.notebook import tqdm as notebook_tqdm
        NOTEBOOK_TQDM_AVAILABLE = True
    except ImportError:
        NOTEBOOK_TQDM_AVAILABLE = False
        notebook_tqdm = None

except ImportError:
    TQDM_AVAILABLE = False
    NOTEBOOK_TQDM_AVAILABLE = False
    tqdm = None
    notebook_tqdm = None

from ..core.base import Monitor


class DetailedMonitor(Monitor):
    """
    Advanced monitor with detailed telemetry support.
    
    Provides comprehensive logging and telemetry for CA training,
    supporting the N_COMPLEX_TELEMETRY hyperparameter.
    
    Examples
    --------
    >>> monitor = DetailedMonitor(
    ...     log_every=10,
    ...     detailed_every=30,  # N_COMPLEX_TELEMETRY
    ...     save_history=True
    ... )
    >>> trainer = Trainer(encoder, sim, fitness, optim, monitor)
    """
    
    def __init__(self, log_every: int = 10, detailed_every: int = 30,
                 save_history: bool = True, history_file: Optional[str] = None):
        """
        Initialize detailed monitor.
        
        Parameters
        ----------
        log_every : int, default=10
            Log basic stats every N generations
        detailed_every : int, default=30
            Log detailed telemetry every N generations (N_COMPLEX_TELEMETRY)
        save_history : bool, default=True
            Whether to save training history
        history_file : str, optional
            File to save history to
        """
        self.log_every = log_every
        self.detailed_every = detailed_every
        self.save_history = save_history
        self.history_file = history_file or "training_history.json"
        
        # Training history
        self.history = {
            'generations': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'diversity': [],
            'accuracy': [],
            'convergence_rate': [],
            'generation_times': [],
            'detailed_telemetry': []
        }
        
        # State tracking
        self.start_time = None
        self.last_best = 0.0
        self.stagnation_count = 0
    
    def update(self, generation: int, scores: np.ndarray, **kwargs) -> None:
        """
        Update monitor with training progress.
        
        Parameters
        ----------
        generation : int
            Current generation number
        scores : np.ndarray
            Fitness scores for current generation
        **kwargs
            Additional data (population, diversity metrics, etc.)
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        # Calculate basic stats
        best_score = np.max(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        accuracy = kwargs.get('accuracy', None)
        
        # Track convergence
        if best_score > self.last_best:
            self.stagnation_count = 0
            self.last_best = best_score
        else:
            self.stagnation_count += 1
        
        # Calculate diversity if population provided
        diversity = kwargs.get('diversity', 0.0)
        if 'population' in kwargs:
            diversity = self._calculate_diversity(kwargs['population'])


        # Store history
        if self.save_history:
            self.history['generations'].append(generation)
            self.history['best_fitness'].append(best_score)
            self.history['mean_fitness'].append(mean_score)
            self.history['std_fitness'].append(std_score)
            self.history['diversity'].append(diversity)
            if accuracy is not None:
                self.history['accuracy'].append(accuracy)
            self.history['convergence_rate'].append(self.stagnation_count)
            self.history['generation_times'].append(time.time() - self.start_time)
        
        # Basic logging
        if generation % self.log_every == 0:
            elapsed = time.time() - self.start_time
            print(f"Gen {generation:4d}: "
                  f"Best={best_score:.4f}, "
                  f"Mean={mean_score:.4f}±{std_score:.4f}, "
                  f"Time={elapsed:.1f}s")
        
        # Detailed telemetry
        if generation % self.detailed_every == 0:
            self._log_detailed_telemetry(generation=generation, scores=scores,
                               diversity=diversity, **kwargs)
    
    def _calculate_diversity(self, population) -> float:
        """Calculate population diversity."""
        try:
            # Programme diversity
            programmes = [genome.programme.code for genome in population]
            unique_programmes = len(set(tuple(p) for p in programmes))
            return unique_programmes / len(population)
        except:
            return 0.0
    
    def _log_detailed_telemetry(self, generation: int, scores: np.ndarray,
                               diversity: float, **kwargs) -> None:
        """Log detailed telemetry information."""

        best_score = np.max(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        median_score = np.median(scores)
        accuracy = kwargs.get('accuracy', None)

        # Performance metrics
        elapsed = time.time() - self.start_time
        gen_per_sec = generation / elapsed if elapsed > 0 else 0

        # Convergence metrics
        improvement_rate = (best_score - self.history['best_fitness'][0]) / generation if generation > 0 else 0

        telemetry = {
            'generation': generation,
            'fitness_stats': {
                'best': best_score,
                'mean': mean_score,
                'std': std_score,
                'median': median_score,
                'range': [np.min(scores), np.max(scores)]
            },
            'population_stats': {
                'diversity': diversity,
                'stagnation_count': self.stagnation_count
            },
            'performance': {
                'elapsed_time': elapsed,
                'generations_per_second': gen_per_sec,
                'improvement_rate': improvement_rate
            },
            'accuracy_stats': {
                'accuracy': accuracy
            }
        }
        
        # Add to history
        if self.save_history:
            self.history['detailed_telemetry'].append(telemetry)
        
        # Print detailed info
        print(f"\n📊 DETAILED TELEMETRY - Generation {generation}")
        print("=" * 50)
        print(f"Fitness: Best={best_score:.6f}, Mean={mean_score:.6f}, Std={std_score:.6f}")
        print(f"Performance: {gen_per_sec:.2f} gen/sec, Improvement={improvement_rate:.6f}/gen")

        # Add best genome accuracy if available
        best_genome_accuracy = 0.0
        if 'best_genome_accuracy' in self.history and len(self.history['best_genome_accuracy']) > 0:
            best_genome_accuracy = self.history['best_genome_accuracy'][-1]
        
        telemetry['accuracy_stats']['best_genome_accuracy'] = best_genome_accuracy
        
        print(f"Accuracy: {accuracy:.2f}% of outputs match targets")
        if best_genome_accuracy > 0:
            print(f"Best Genome Test Accuracy: {best_genome_accuracy:.2f}%")
        
        # Convergence analysis
        if self.stagnation_count > 50:
            print("⚠️  Warning: Population may have converged (50+ generations without improvement)")

        if diversity < 0.1:
            print("⚠️  Warning: Low diversity detected - consider increasing mutation or immigrants")

        # Update convergence analysis
        if best_genome_accuracy >= 100.0:
            print("🎉 Perfect accuracy achieved! Best genome solves all test cases.")
        elif best_genome_accuracy >= 90.0:
            print("🔥 Excellent accuracy! Best genome very close to perfect solution.")

        print("=" * 50 + "\n")
    
    def save_history(self, filename: Optional[str] = None) -> None:
        """Save training history to file."""
        if not self.save_history:
            return
        
        filename = filename or self.history_file
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            history_json = {}
            for key, value in self.history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        history_json[key] = [v.tolist() for v in value]
                    else:
                        history_json[key] = value
                else:
                    history_json[key] = value
            
            with open(filename, 'w') as f:
                json.dump(history_json, f, indent=2)
            
            print(f"💾 Training history saved to {filename}")
        
        except Exception as e:
            print(f"⚠️  Failed to save history: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.history['generations']:
            return {}

        summary = {
            'total_generations': len(self.history['generations']),
            'final_best_fitness': self.history['best_fitness'][-1],
            'max_fitness_achieved': max(self.history['best_fitness']),
            'final_diversity': self.history['diversity'][-1] if self.history['diversity'] else 0,
            'total_time': self.history['generation_times'][-1] if self.history['generation_times'] else 0,
            'convergence_generation': self._find_convergence_point(),
            'stagnation_periods': self._count_stagnation_periods()
        }

        # Add accuracy information if available
        if self.history['accuracy']:
            summary.update({
                'final_accuracy': self.history['accuracy'][-1],
                'max_accuracy_achieved': max(self.history['accuracy']),
                'accuracy_convergence_generation': self._find_accuracy_convergence_point()
            })

        return summary
    
    def _find_convergence_point(self) -> Optional[int]:
        """Find the generation where fitness stopped improving significantly."""
        if len(self.history['best_fitness']) < 10:
            return None
        
        # Look for last significant improvement (>1% increase)
        best_fitness = self.history['best_fitness']
        for i in range(len(best_fitness) - 1, 0, -1):
            if best_fitness[i] > best_fitness[i-1] * 1.01:
                return self.history['generations'][i]
        
        return None

    def _find_accuracy_convergence_point(self) -> Optional[int]:
        """Find the generation where accuracy reached 95% or higher."""
        if not self.history['accuracy'] or len(self.history['accuracy']) < 1:
            return None

        for i, accuracy in enumerate(self.history['accuracy']):
            if accuracy >= 95.0:
                return self.history['generations'][i] if i < len(self.history['generations']) else i

        return None

    def _count_stagnation_periods(self) -> int:
        """Count number of stagnation periods (10+ generations without improvement)."""
        if len(self.history['convergence_rate']) < 10:
            return 0
        
        stagnation_periods = 0
        in_stagnation = False
        
        for count in self.history['convergence_rate']:
            if count >= 10 and not in_stagnation:
                stagnation_periods += 1
                in_stagnation = True
            elif count < 10:
                in_stagnation = False
        
        return stagnation_periods


class ProgressMonitor(Monitor):
    """Simple progress monitor with progress bar-like output."""
    
    def __init__(self, target_fitness: float = 1.0, log_every: int = 10):
        """Initialize progress monitor."""
        self.target_fitness = target_fitness
        self.log_every = log_every
        self.start_time = time.time()
    
    def update(self, generation: int, scores: np.ndarray, **kwargs) -> None:
        """Update progress display."""
        if generation % self.log_every == 0:
            best_score = np.max(scores)
            progress = min(100, (best_score / self.target_fitness) * 100)
            elapsed = time.time() - self.start_time
            
            # Simple progress bar
            bar_length = 20
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"Gen {generation:4d} [{bar}] {progress:5.1f}% "
                  f"(Best: {best_score:.4f}, Time: {elapsed:.1f}s)")


class TqdmMonitor(Monitor):
    """Monitor with tqdm progress bar optimized for Jupyter notebooks."""

    def __init__(self, total_generations: int, update_every: int = 1, force_notebook: bool = None):
        """
        Initialize tqdm monitor.

        Parameters
        ----------
        total_generations : int
            Total number of generations for progress bar
        update_every : int, default=1
            Update progress bar every N generations
        force_notebook : bool, optional
            Force use of notebook tqdm. If None, auto-detect environment
        """
        if not TQDM_AVAILABLE:
            raise ImportError("tqdm is required for TqdmMonitor. Install with: pip install tqdm")

        self.total_generations = total_generations
        self.update_every = update_every
        self.pbar = None
        self.last_generation = -1
        self.force_notebook = force_notebook

        # Track best fitness for progress description
        self.best_fitness = 0.0
        self.mean_fitness = 0.0

        # Determine which tqdm to use
        self.tqdm_class = self._get_tqdm_class()

    def _get_tqdm_class(self):
        """Determine the best tqdm class to use."""
        # If explicitly forced
        if self.force_notebook is True:
            if NOTEBOOK_TQDM_AVAILABLE:
                from tqdm.notebook import tqdm as notebook_tqdm
                return notebook_tqdm
            else:
                print("⚠️  Notebook tqdm requested but not available, falling back to auto")
                return tqdm
        elif self.force_notebook is False:
            return tqdm

        # Auto-detect environment
        try:
            # Check if we're in Jupyter
            from IPython import get_ipython
            if get_ipython() is not None and NOTEBOOK_TQDM_AVAILABLE:
                # We're in IPython/Jupyter and notebook tqdm is available
                from tqdm.notebook import tqdm as notebook_tqdm
                return notebook_tqdm
        except ImportError:
            pass

        # Fall back to regular tqdm
        return tqdm

    def update(self, generation: int, scores: np.ndarray, **kwargs) -> None:
        """Update progress bar."""
        if self.pbar is None:
            # Initialize progress bar on first update using the determined tqdm class
            self.pbar = self.tqdm_class(
                total=self.total_generations,
                desc="Training",
                unit="gen",
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )

        # Update fitness tracking
        self.best_fitness = np.max(scores)
        self.mean_fitness = np.mean(scores)

        # Calculate how many steps to advance
        steps_to_advance = generation - self.last_generation
        if steps_to_advance > 0:
            self.pbar.update(steps_to_advance)
            self.last_generation = generation

        # Update description with current stats
        if generation % self.update_every == 0 or generation == self.total_generations - 1:
            postfix = f"Best: {self.best_fitness:.4f}, Mean: {self.mean_fitness:.4f}"
            self.pbar.set_postfix_str(postfix)

        # Close progress bar when training is complete
        if generation >= self.total_generations - 1:
            if self.pbar:
                self.pbar.close()

    def close(self):
        """Manually close the progress bar."""
        if self.pbar:
            self.pbar.close()


class JupyterMonitor(Monitor):
    """Monitor specifically optimized for Jupyter notebooks using tqdm.notebook."""

    def __init__(self, total_generations: int, update_every: int = 1):
        """
        Initialize Jupyter-optimized monitor.

        Parameters
        ----------
        total_generations : int
            Total number of generations for progress bar
        update_every : int, default=1
            Update progress bar every N generations
        """
        try:
            from tqdm.notebook import tqdm as notebook_tqdm
            self.tqdm_class = notebook_tqdm
        except ImportError:
            raise ImportError(
                "tqdm.notebook is required for JupyterMonitor. "
                "Install with: pip install tqdm ipywidgets"
            )

        self.total_generations = total_generations
        self.update_every = update_every
        self.pbar = None
        self.last_generation = -1

        # Track best fitness for progress description
        self.best_fitness = 0.0
        self.mean_fitness = 0.0

    def update(self, generation: int, scores: np.ndarray, **kwargs) -> None:
        """Update progress bar."""
        if self.pbar is None:
            # Initialize progress bar on first update
            self.pbar = self.tqdm_class(
                total=self.total_generations,
                desc="🚀 Training",
                unit="gen",
                leave=True
            )

        # Update fitness tracking
        self.best_fitness = np.max(scores)
        self.mean_fitness = np.mean(scores)

        # Calculate how many steps to advance
        steps_to_advance = generation - self.last_generation
        if steps_to_advance > 0:
            self.pbar.update(steps_to_advance)
            self.last_generation = generation

        # Update description with current stats
        if generation % self.update_every == 0 or generation == self.total_generations - 1:
            postfix = f"Best: {self.best_fitness:.4f}, Mean: {self.mean_fitness:.4f}"
            self.pbar.set_postfix_str(postfix)

        # Close progress bar when training is complete
        if generation >= self.total_generations - 1:
            if self.pbar:
                self.pbar.close()

    def close(self):
        """Manually close the progress bar."""
        if self.pbar:
            self.pbar.close()


class CombinedMonitor(Monitor):
    """Monitor that combines multiple monitors (e.g., tqdm + detailed logging)."""

    def __init__(self, *monitors):
        """
        Initialize combined monitor.

        Parameters
        ----------
        *monitors : Monitor
            Multiple monitor instances to combine
        """
        self.monitors = monitors

    def update(self, generation: int, scores: np.ndarray, **kwargs) -> None:
        """Update all monitors."""
        for monitor in self.monitors:
            monitor.update(generation, scores, **kwargs)

    def close(self):
        """Close all monitors that support it."""
        for monitor in self.monitors:
            if hasattr(monitor, 'close'):
                monitor.close()


class SilentMonitor(Monitor):
    """Silent monitor that only tracks history without printing."""

    def __init__(self):
        """Initialize silent monitor."""
        self.history = []

    def update(self, generation: int, scores: np.ndarray, **kwargs) -> None:
        """Silently record progress."""
        self.history.append({
            'generation': generation,
            'best_fitness': np.max(scores),
            'mean_fitness': np.mean(scores),
            'std_fitness': np.std(scores)
        })
