"""
Optimizer implementations for cellular automata evolution.

This module provides optimizers that implement search strategies
for evolving CA genomes.
"""

import numpy as np
from typing import List, Optional
import math

from ..genome import Genome, create_random_genome
from ..core.state import StateModel


class GAOptimizer:
    """
    Genetic Algorithm optimizer with tournament selection and elitism.
    
    Implements a genetic algorithm with:
    - Tournament selection
    - Elitism (preserve best individuals)
    - Random immigrants (maintain diversity)
    - Configurable mutation and crossover rates
    
    Examples
    --------
    >>> from emergent_models.core import StateModel
    >>> state = StateModel([0,1,2,3])
    >>> optimizer = GAOptimizer(pop_size=100, state=state, prog_len=10)
    >>> 
    >>> # Initialize population
    >>> population = optimizer.ask()
    >>> 
    >>> # Evaluate fitness (dummy scores)
    >>> scores = np.random.random(100)
    >>> optimizer.tell(scores)
    >>> 
    >>> # Get next generation
    >>> next_population = optimizer.ask()
    """
    
    def __init__(self, pop_size: int, state: StateModel, prog_len: int = 10,
                 mutation_rate: float = 0.03, prog_mutation_rate: float = 0.08,
                 elite_fraction: float = 0.1, tournament_size: int = 3,
                 random_immigrant_rate: float = 0.2, prog_sparsity: float = 0.01):
        """
        Initialize genetic algorithm optimizer.
        
        Parameters
        ----------
        pop_size : int
            Population size
        state : StateModel
            State model for creating genomes
        prog_len : int, default=10
            Programme length for genomes
        mutation_rate : float, default=0.03
            Mutation rate for rules
        prog_mutation_rate : float, default=0.08
            Mutation rate for programmes
        elite_fraction : float, default=0.1
            Fraction of population to preserve as elite
        tournament_size : int, default=3
            Tournament size for selection
        random_immigrant_rate : float, default=0.2
            Fraction of population to replace with random immigrants
        prog_sparsity : float, default=0.6
            Target sparsity for random programmes
        """
        self.pop_size = pop_size
        self.state = state
        self.prog_len = prog_len
        self.mutation_rate = mutation_rate
        self.prog_mutation_rate = prog_mutation_rate
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size
        self.random_immigrant_rate = random_immigrant_rate
        self.prog_sparsity = prog_sparsity
        
        # Calculate derived parameters
        self.n_elite = max(1, int(pop_size * elite_fraction))
        self.n_immigrants = int(pop_size * random_immigrant_rate)
        
        # Initialize state
        self.population: List[Genome] = []
        self.fitness_scores: List[float] = []
        self.generation = 0
        self.rng = np.random.default_rng()
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize random population."""
        self.population = []
        for _ in range(self.pop_size):
            genome = create_random_genome(
                self.state, self.prog_len, self.prog_sparsity, self.rng
            )
            self.population.append(genome)
        
        self.fitness_scores = [0.0] * self.pop_size
    
    def ask(self) -> List[Genome]:
        """
        Get current population for evaluation.
        
        Returns
        -------
        List[Genome]
            Current population to evaluate
        """
        return self.population.copy()
    
    def tell(self, scores: np.ndarray) -> None:
        """
        Update population with fitness scores and evolve.
        
        Parameters
        ----------
        scores : np.ndarray
            Fitness scores for current population
        """
        if len(scores) != len(self.population):
            raise ValueError("Number of scores must match population size")
        
        self.fitness_scores = scores.tolist()
        
        # Update fitness in genomes
        for genome, fitness in zip(self.population, self.fitness_scores):
            genome.fitness = fitness
        
        # Evolve to next generation
        self._evolve()
        self.generation += 1
    
    def _evolve(self):
        """Evolve population to next generation."""
        fitness_array = np.array(self.fitness_scores)
        sorted_indices = np.argsort(fitness_array)[::-1]

        next_population = []
        
        # Add elite individuals directly
        for i in range(self.n_elite):
            elite_index = sorted_indices[i]
            next_population.append(self.population[elite_index].copy())
        
        # Generate offspring through tournament selection and crossover
        while len(next_population) < self.pop_size:
            parent1 = self._tournament_selection(self.population)
            parent2 = self._tournament_selection(self.population)
            
            child = parent1.crossover(parent2, self.rng)
            child = self._mutate(child)
            
            next_population.append(child)
        
        # Replace some individuals with random immigrants
        for _ in range(self.n_immigrants):
            if len(next_population) > self.n_elite:
                idx = self.rng.integers(self.n_elite, len(next_population))
                immigrant = create_random_genome(
                    self.state, self.prog_len, self.prog_sparsity, self.rng
                )
                next_population[idx] = immigrant
        
        # Ensure exact population size
        self.population = next_population[:self.pop_size]
    
    def _tournament_selection(self, population: List[Genome]) -> Genome:
        """Select individual using tournament selection."""
        tournament_indices = self.rng.choice(
            len(population), size=self.tournament_size, replace=False
        )
        
        best_idx = tournament_indices[0]
        best_fitness = population[best_idx].fitness
        
        for idx in tournament_indices[1:]:
            if population[idx].fitness > best_fitness:
                best_fitness = population[idx].fitness
                best_idx = idx
        
        return population[best_idx]
    
    def _mutate(self, genome: Genome) -> Genome:
        """Apply mutation to genome."""
        return genome.mutate(self.mutation_rate, self.prog_mutation_rate, self.rng)
    
    def best_genome(self) -> Genome:
        """Get the best genome from current population."""
        if not self.population:
            return None

        best_idx = int(np.argmax(self.fitness_scores))
        return self.population[best_idx].copy()
    
    def best_fitness(self) -> float:
        """Get the best fitness from current population."""
        if not self.fitness_scores:
            return 0.0
        return max(self.fitness_scores)
    
    def mean_fitness(self) -> float:
        """Get the mean fitness of current population."""
        if not self.fitness_scores:
            return 0.0
        return np.mean(self.fitness_scores)
    
    def fitness_stats(self) -> dict:
        """Get fitness statistics for current population."""
        if not self.fitness_scores:
            return {"best": 0.0, "mean": 0.0, "std": 0.0}
        
        scores = np.array(self.fitness_scores)
        return {
            "best": np.max(scores),
            "mean": np.mean(scores),
            "std": np.std(scores)
        }
    
    def __repr__(self) -> str:
        return (f"GAOptimizer(pop_size={self.pop_size}, "
                f"generation={self.generation}, "
                f"best_fitness={self.best_fitness():.4f})")


class RandomSearchOptimizer:
    """
    Random search optimizer for baseline comparison.
    
    Simply generates random genomes each generation.
    """
    
    def __init__(self, pop_size: int, state: StateModel, prog_len: int = 10,
                 prog_sparsity: float = 0.3):
        """Initialize random search optimizer."""
        self.pop_size = pop_size
        self.state = state
        self.prog_len = prog_len
        self.prog_sparsity = prog_sparsity
        self.generation = 0
        self.rng = np.random.default_rng()
        
        self.population: List[Genome] = []
        self.fitness_scores: List[float] = []
        
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize random population."""
        self.population = []
        for _ in range(self.pop_size):
            genome = create_random_genome(
                self.state, self.prog_len, self.prog_sparsity, self.rng
            )
            self.population.append(genome)
        
        self.fitness_scores = [0.0] * self.pop_size
    
    def ask(self) -> List[Genome]:
        """Get current population for evaluation."""
        return self.population.copy()
    
    def tell(self, scores: np.ndarray) -> None:
        """Update with scores and generate new random population."""
        self.fitness_scores = scores.tolist()
        
        # Generate completely new random population
        self._initialize_population()
        self.generation += 1
    
    def best_genome(self) -> Genome:
        """Get the best genome from current population."""
        if not self.population:
            return None

        best_idx = int(np.argmax(self.fitness_scores))
        return self.population[best_idx].copy()
    
    def best_fitness(self) -> float:
        """Get the best fitness from current population."""
        if not self.fitness_scores:
            return 0.0
        return max(self.fitness_scores)
    
    def fitness_stats(self) -> dict:
        """Get fitness statistics for current population."""
        if not self.fitness_scores:
            return {"best": 0.0, "mean": 0.0, "std": 0.0}
        
        scores = np.array(self.fitness_scores)
        return {
            "best": np.max(scores),
            "mean": np.mean(scores),
            "std": np.std(scores)
        }
