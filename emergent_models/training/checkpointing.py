from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any

from ..genome import Genome


def save_genome(genome: Genome, filepath: str) -> None:
    """Save genome to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON for human readability
    if filepath.suffix != '.json':
        filepath = filepath.with_suffix('.json')

    data = {
        'type': 'Genome',
        'rule_table': genome.rule.table.tolist(),
        'programme_code': genome.programme.code.tolist(),
        'fitness': float(genome.fitness)
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_genome(filepath: str) -> Genome:
    """Load genome from file."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Genome file not found: {filepath}")

    # Load from JSON format
    with open(filepath, 'r') as f:
        data = json.load(f)

    if data.get('type') != 'Genome':
        raise ValueError(f"Invalid genome type: {data.get('type')}")

    # Reconstruct genome (this would need proper state model setup)
    # For now, just return a basic structure
    from ..rules.ruleset import RuleSet
    from ..rules.programme import Programme
    from ..core.state import StateModel

    state = StateModel([0, 1, 2, 3])  # Default EM-4/3 state
    rule_table = np.array(data['rule_table'], dtype=np.uint8)
    programme_code = np.array(data['programme_code'], dtype=np.uint8)

    rule = RuleSet(rule_table, state)
    programme = Programme(programme_code, state)
    genome = Genome(rule, programme)
    genome.fitness = data.get('fitness', 0.0)

    return genome





def save_population(
    population: list,
    directory: Union[str, Path],
    prefix: str = "genome"
) -> None:
    """Save an entire population to a directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for i, genome in enumerate(population):
        filename = f"{prefix}_{i:04d}.json"
        save_genome(genome, directory / filename)

    # Save population metadata
    metadata = {
        'population_size': len(population),
        'genome_type': type(population[0]).__name__ if population else 'Unknown'
    }

    with open(directory / 'population_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def load_population(directory: Union[str, Path]) -> list:
    """Load an entire population from a directory."""
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Population directory not found: {directory}")

    # Load all genome files
    genome_files = sorted(directory.glob("genome_*.json"))

    if not genome_files:
        raise ValueError(f"No genome files found in {directory}")

    population = []
    for genome_file in genome_files:
        genome = load_genome(genome_file)
        population.append(genome)

    return population


def save_training_checkpoint(
    trainer,
    filepath: Union[str, Path],
    include_population: bool = True
) -> None:
    """Save a complete training checkpoint."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = {
        'generation': trainer.optimizer.generation,
        'best_fitness': trainer.best_fitness,
        'history': trainer.history,
        'optimizer_config': {
            'population_size': trainer.optimizer.population_size,
            'mutation_rate': trainer.optimizer.mutation_rate,
            'elite_fraction': trainer.optimizer.elite_fraction,
            'tournament_size': trainer.optimizer.tournament_size
        }
    }

    # Save best genome separately
    if trainer.best_genome is not None:
        best_genome_path = filepath.with_suffix('.best_genome.json')
        save_genome(trainer.best_genome, best_genome_path)
        checkpoint_data['best_genome_file'] = str(best_genome_path)

    # Save population if requested
    if include_population and trainer.optimizer.population:
        population_dir = filepath.parent / f"{filepath.stem}_population"
        save_population(trainer.optimizer.population, population_dir)
        checkpoint_data['population_directory'] = str(population_dir)

    # Save checkpoint metadata
    with open(filepath, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def load_training_checkpoint(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load a training checkpoint."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    with open(filepath, 'r') as f:
        checkpoint_data = json.load(f)

    # Load best genome if available
    if 'best_genome_file' in checkpoint_data:
        best_genome_path = Path(checkpoint_data['best_genome_file'])
        if best_genome_path.exists():
            checkpoint_data['best_genome'] = load_genome(best_genome_path)

    # Load population if available
    if 'population_directory' in checkpoint_data:
        population_dir = Path(checkpoint_data['population_directory'])
        if population_dir.exists():
            checkpoint_data['population'] = load_population(population_dir)

    return checkpoint_data
