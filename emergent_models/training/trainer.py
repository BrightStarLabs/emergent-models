"""
Trainer implementation with fused pipeline for maximum performance.

This module provides the Trainer class that orchestrates all components
and builds fused Numba kernels for optimal performance.
"""

import numpy as np
from typing import List, Optional
import time

import numba as nb

from ..core.base import Encoder, FitnessFn, Monitor, ConsoleLogger
from ..simulation.simulator import Simulator
from ..training.optimizer import GAOptimizer
from ..genome import Genome


class Trainer:
    """
    High-level trainer that orchestrates all components with fused pipeline.
    
    The trainer knows about all components and builds a single fused Numba
    kernel that inlines encode → simulate → decode for maximum performance.
    
    Examples
    --------
    >>> from emergent_models.core import StateModel, Tape1D
    >>> from emergent_models.encoders import Em43Encoder
    >>> from emergent_models.simulation import Simulator
    >>> from emergent_models.training import GAOptimizer, AbsoluteDifferenceFitness
    >>> 
    >>> # Setup components
    >>> state = StateModel([0,1,2,3])
    >>> space = Tape1D(200, radius=1)
    >>> encoder = Em43Encoder(state, space)
    >>> sim = Simulator(state, space, max_steps=256, halt_thresh=0.5)
    >>> fitness = AbsoluteDifferenceFitness()
    >>> optim = GAOptimizer(pop_size=100, state=state, prog_len=10)
    >>> 
    >>> # Create trainer
    >>> trainer = Trainer(encoder, sim, fitness, optim)
    >>> 
    >>> # Train
    >>> trainer.fit(inputs=range(1, 11), generations=50)
    """
    
    def __init__(self, encoder: Encoder, simulator: Simulator,
                 fitness_fn: FitnessFn, optimizer: GAOptimizer,
                 monitor: Optional[Monitor] = None):
        """
        Initialize trainer with all components.

        Parameters
        ----------
        encoder : Encoder
            Encoder for converting data to/from CA tapes
        simulator : Simulator
            Simulator for CA evolution
        fitness_fn : FitnessFn
            Fitness function for evaluation
        optimizer : GAOptimizer
            Optimizer for population evolution
        monitor : Monitor, optional
            Monitor for logging/tracking progress
        """
        self.encoder = encoder
        self.simulator = simulator
        self.fitness_fn = fitness_fn
        self.optimizer = optimizer
        self.monitor = monitor or ConsoleLogger()

        # Pre-allocate evaluation buffers for maximum performance
        self._init_evaluation_buffers()

        # Store last decoded outputs for accuracy tracking
        self._last_decoded_outputs = None

        # Training history
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'generation_times': []
        }

    def _init_evaluation_buffers(self):
        """Initialize pre-allocated buffers for population evaluation."""
        # Get buffer sizes from optimizer and components
        pop_size = self.optimizer.pop_size
        prog_len = self.optimizer.prog_len

        # Determine rule table size (assumes 4-state, 3-neighborhood = 64 entries)
        n_states = self.optimizer.state.n_states
        table_size = n_states ** 3

        # Pre-allocate population data buffers
        self._rule_buffer = np.zeros((pop_size, table_size), dtype=np.uint8)
        self._prog_buffer = np.zeros((pop_size, prog_len), dtype=np.uint8)
        self._fitness_buffer = np.zeros(pop_size, dtype=np.float32)
        self._sparsity_buffer = np.zeros(pop_size, dtype=np.float32)

        print(f"🚀 Pre-allocated evaluation buffers: pop_size={pop_size}, "
              f"table_size={table_size}, prog_len={prog_len}")

    def evaluate_population(self, genomes: List[Genome],
                          inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Evaluate entire population using optimized fused kernel.
        (P, ProgLen) + (B, 1) -> (P, )

        Parameters
        ----------
        genomes : List[Genome]
            Population of genomes to evaluate (P, ProgLen)
        inputs : np.ndarray
            Input values to test (B, 1)
        targets : np.ndarray
            Target values to compare against (B, 1)

        Returns
        -------
        np.ndarray
            Fitness scores for each genome
        """
        pop_size = len(genomes)

        # Fast batch extraction using pre-allocated buffers
        self._extract_population_data_fast(genomes)

        encoded_inputs = self.encoder.encode_population(self._prog_buffer[:pop_size], inputs)  # (P, ProgLen) + (B, 1) -> (P, B, T)
            
        final_state = self.simulator.run_batch(encoded_inputs, self._rule_buffer[:pop_size])  # (P, B, T) + (P, TableSize) -> (P, B, T)
        decoded_outputs = self.encoder.decode_population(final_state, self._prog_buffer[:pop_size].shape[1])  # (P, B, T) + ProgLen -> (P, B, 1)

        decoded_outputs = decoded_outputs.squeeze()  # (P, B, 1) -> (P, B)
        targets = targets.squeeze()  # (B, 1) -> (B,)

        fitness_scores = self.fitness_fn(decoded_outputs, targets) # (P, B) + (B, ) -> (P,)

        # Store outputs for accuracy tracking
        self._last_decoded_outputs = decoded_outputs

        return fitness_scores

    def _extract_population_data_fast(self, genomes: List[Genome]) -> None:
        """
        Fast extraction of population data into pre-allocated buffers.

        This method optimizes the data extraction bottleneck by minimizing
        Python loops and using vectorized operations where possible.
        """
        pop_size = len(genomes)

        # Extract rule tables and programmes in optimized loop
        for i in range(pop_size):
            # Direct array assignment (faster than individual element access)
            np.copyto(self._rule_buffer[i], genomes[i].rule.table)
            np.copyto(self._prog_buffer[i], genomes[i].programme.code)
    
    def fit(self, inputs: np.ndarray, targets: np.ndarray, generations: int = 100,
            test_inputs: Optional[np.ndarray] = None,
            test_targets: Optional[np.ndarray] = None,
            early_stopping_threshold: Optional[float] = None,
            checkpoint_every: Optional[int] = None,
            checkpoint_path: str = "checkpoints/",
            use_tqdm: bool = True) -> dict:
        """
        Train the population using genetic algorithm.

        Parameters
        ----------
        inputs : np.ndarray
            Input values for training
        generations : int, default=100
            Number of generations to train
        early_stopping_threshold : float, optional
            Stop training when best fitness reaches this threshold
        checkpoint_every : int, optional
            Save checkpoint every N generations (CHECK_EVERY hyperparameter)
        checkpoint_path : str, default="checkpoints/"
            Directory to save checkpoints
        use_tqdm : bool, default=True
            Whether to show tqdm progress bar (useful for Jupyter notebooks)

        Returns
        -------
        dict
            Training results including best genome and history
        """
        inputs = np.asarray(inputs, dtype=np.int64)

        print(f"Starting training for {generations} generations...")
        print(f"Population size: {self.optimizer.pop_size}")
        if checkpoint_every:
            print(f"Checkpointing every {checkpoint_every} generations to {checkpoint_path}")

        # Setup checkpointing
        if checkpoint_every:
            import os
            os.makedirs(checkpoint_path, exist_ok=True)

        # Setup tqdm progress bar if requested
        tqdm_monitor = None
        if use_tqdm:
            try:
                from .monitor import TqdmMonitor, CombinedMonitor
                tqdm_monitor = TqdmMonitor(generations)

                # Combine tqdm with existing monitor
                if self.monitor:
                    combined_monitor = CombinedMonitor(tqdm_monitor, self.monitor)
                    original_monitor = self.monitor
                    self.monitor = combined_monitor
                else:
                    self.monitor = tqdm_monitor
                    original_monitor = None
            except ImportError:
                print("⚠️  tqdm not available, falling back to console output")
                tqdm_monitor = None
                original_monitor = None
        else:
            original_monitor = None

        gen = 0  # Initialize gen outside the loop for proper scope
        for gen in range(generations):
            start_time = time.time()

            # Get current population
            population = self.optimizer.ask()

            # Evaluate population
            fitness_scores = self.evaluate_population(population, inputs, targets)

            # Update optimizer
            self.optimizer.tell(fitness_scores)

            # Record history
            gen_time = time.time() - start_time
            stats = self.optimizer.fitness_stats()

            self.history['best_fitness'].append(stats['best'])
            self.history['mean_fitness'].append(stats['mean'])
            self.history['std_fitness'].append(stats['std'])
            self.history['generation_times'].append(gen_time)

            # If test data not provided, use training data
            if test_inputs is None:
                test_inputs = inputs
            if test_targets is None and targets is not None:
                test_targets = targets
            
            accuracy = self._calculate_accuracy(test_inputs, test_targets)

            # Update monitor with outputs and targets for accuracy calculation
            population = self.optimizer.ask()  # Get population for diversity calculation
            self.monitor.update(gen, fitness_scores, 
                      population=population,
                      outputs=self._last_decoded_outputs, 
                      targets=targets,
                      accuracy=accuracy,
                      trainer=self,
                      **stats)

            # Check for early stopping based on best genome accuracy
            if hasattr(self.monitor, 'history') and 'best_genome_accuracy' in self.monitor.history:
                best_genome_accuracy = self.monitor.history['best_genome_accuracy'][-1]
                if best_genome_accuracy >= early_stopping_threshold:
                    print(f"🎯 Early stopping at generation {gen}: "
                        f"best genome achieved {best_genome_accuracy:.2f}% accuracy on test set")
                    break
                
            # Save checkpoint
            if checkpoint_every and gen % checkpoint_every == 0 and gen > 0:
                self._save_checkpoint(gen, checkpoint_path)
                print(f"🔍 DEBUG: Saved checkpoint at generation {gen}")

            # Check early stopping
            if (early_stopping_threshold is not None and
                stats['best'] >= early_stopping_threshold):
                print(f"Early stopping at generation {gen}: "
                      f"fitness {stats['best']:.4f} >= {early_stopping_threshold}")
                break

        print(f"🔍 DEBUG: Training loop completed. Final gen = {gen}, requested generations = {generations}")
        
        # Close tqdm progress bar if used
        if tqdm_monitor and hasattr(self.monitor, 'close'):
            self.monitor.close()

        # Save final checkpoint
        if checkpoint_every:
            print(f"🔍 DEBUG: Saving final checkpoint. gen = {gen}, generations = {generations}")
            self._save_checkpoint(gen, checkpoint_path, final=True)

        # Save monitor history if available
        if hasattr(self.monitor, 'save_history') and callable(self.monitor.save_history):
            self.monitor.save_history()

        # Return results
        results = {
            'best_genome': self.optimizer.best_genome(),
            'best_fitness': self.optimizer.best_fitness(),
            'history': self.history,
            'final_generation': gen
        }

        # Add monitor summary if available
        if hasattr(self.monitor, 'get_summary'):
            results['monitor_summary'] = self.monitor.get_summary()

        return results
    def _calculate_accuracy(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Calculate accuracy of best genome on given inputs and targets."""
        best_genome = self.optimizer.best_genome()
        decoded_outputs = self.evaluate_single_genome(best_genome, inputs)
        
        # Make sure shapes match for comparison
        if decoded_outputs.ndim != targets.ndim:
            if decoded_outputs.ndim > targets.ndim:
                targets = targets.reshape(decoded_outputs.shape)
            else:
                decoded_outputs = decoded_outputs.reshape(targets.shape)
            
        accuracy = np.mean(decoded_outputs == targets) * 100.0  # Convert to percentage
        return accuracy

    def _save_checkpoint(self, generation: int, checkpoint_path: str, final: bool = False) -> None:
        """Save training checkpoint."""
        try:
            import pickle
            import os

            checkpoint_name = f"final_checkpoint.pkl" if final else f"checkpoint_gen_{generation}.pkl"
            checkpoint_file = os.path.join(checkpoint_path, checkpoint_name)

            checkpoint_data = {
                'generation': generation,
                'optimizer_state': {
                    'population': self.optimizer.population,
                    'fitness_scores': self.optimizer.fitness_scores,
                    'generation': self.optimizer.generation
                },
                'best_genome': self.optimizer.best_genome(),
                'best_fitness': self.optimizer.best_fitness(),
                'history': self.history
            }

            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

            print(f"💾 Checkpoint saved: {checkpoint_file}")

        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_file: str) -> dict:
        """Load training checkpoint."""
        try:
            import pickle

            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # Restore optimizer state
            self.optimizer.population = checkpoint_data['optimizer_state']['population']
            self.optimizer.fitness_scores = checkpoint_data['optimizer_state']['fitness_scores']
            self.optimizer.generation = checkpoint_data['optimizer_state']['generation']

            # Restore history
            self.history = checkpoint_data['history']

            print(f"✅ Checkpoint loaded: {checkpoint_file}")
            print(f"   Resumed from generation {checkpoint_data['generation']}")
            print(f"   Best fitness: {checkpoint_data['best_fitness']:.4f}")

            return checkpoint_data

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return {}

    def evaluate_single_genome(self, genome, inputs):
        """
        Evaluate a single genome on multiple inputs.
        
        Parameters
        ----------
        genome : Genome
            Genome to evaluate
        inputs : np.ndarray
            Input values to test (B, )
        
        Returns
        -------
        np.ndarray
            Decoded outputs for each input (B, )
        """
        # Extract genome data
        rule_table = genome.rule.table
        programme = genome.programme.code
        
        batch_size = len(inputs)
        programmes_batch = programme.reshape(1, -1) # (1, L)
        
        # Encode inputs using encode_population
        encoded_inputs = self.encoder.encode_population(programmes_batch, inputs)
        
        rule_tables_batch = rule_table.reshape(1, -1) # (1, TableSize)
        
        # Run simulation using run_batch
        final_states = self.simulator.run_batch(encoded_inputs, rule_tables_batch)
        
        # Decode outputs
        decoded_outputs = self.encoder.decode_population(final_states, len(programme))
        
        return decoded_outputs.squeeze()

