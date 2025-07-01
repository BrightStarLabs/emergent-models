#!/usr/bin/env python3
"""
Training Playground
=================
Play with parameters to see how they affect training.
If it doesn't work, try different parameters and see how they affect training.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse  # Add argparse for CLI arguments

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from emergent_models.core.state import StateModel
from emergent_models.core.space_model import Tape1D
from emergent_models.encoders.em43 import Em43Encoder
from emergent_models.simulation.simulator import Simulator
from emergent_models.training import AbsoluteDifferenceFitness
from emergent_models.training.optimizer import GAOptimizer
from emergent_models.training import Trainer
from emergent_models.core.base import ConsoleLogger


def parse_arguments():
    """Parse command line arguments for hyperparameters."""
    parser = argparse.ArgumentParser(description='Training Debugger for Emergent Models')
    
    # Optimizer parameters
    parser.add_argument('--pop-size', type=int, default=50,
                        help='Population size (default: 50)')
    parser.add_argument('--prog-len', type=int, default=8,
                        help='Programme length (default: 8)')
    parser.add_argument('--mutation-rate', type=float, default=0.01,
                        help='Rule mutation rate (default: 0.01)')
    parser.add_argument('--prog-mutation-rate', type=float, default=0.05,
                        help='Programme mutation rate (default: 0.05)')
    parser.add_argument('--elite-fraction', type=float, default=0.2,
                        help='Elite fraction (default: 0.2)')
    parser.add_argument('--random-immigrant-rate', type=float, default=0.01,
                        help='Random immigrant rate (default: 0.1)')
    
    # Simulator parameters
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Maximum simulation steps (default: 50)')
    parser.add_argument('--halt-thresh', type=float, default=0.5,
                        help='Halting threshold (default: 0.5)')
    parser.add_argument('--tape-length', type=int, default=100,
                        help='Tape length (default: 100)')
    
    # Training parameters
    parser.add_argument('--generations', type=int, default=10,
                        help='Number of generations to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=30,
                        help='Batch size for training (default: 30)')
    parser.add_argument('--input-min', type=int, default=1,
                        help='Minimum input value (default: 1)')
    parser.add_argument('--input-max', type=int, default=30,
                        help='Maximum input value (default: 30)')
    parser.add_argument('--target-fn', type=str, default='double',
                        choices=['double', 'identity', 'square'],
                        help='Target function (default: double)')
    parser.add_argument('--continuous-fitness', action='store_true',
                        help='Use continuous fitness (default: True)')
    
    # Visualization parameters
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting (default: False)')
    
    return parser.parse_args()


def get_target_function(name):
    """Get target function by name."""
    if name == 'double':
        return lambda x: 2 * x
    elif name == 'identity':
        return lambda x: x
    elif name == 'square':
        return lambda x: x * x
    else:
        raise ValueError(f"Unknown target function: {name}")


def analyze_population_diversity(population):
    """Analyze diversity in the population."""
    
    print("🔍 POPULATION DIVERSITY ANALYSIS")
    print("=" * 40)
    
    # Analyze programme diversity
    programmes = [genome.programme.code for genome in population]
    unique_programmes = len(set(tuple(p) for p in programmes))
    
    print(f"Population size: {len(population)}")
    print(f"Unique programmes: {unique_programmes}")
    print(f"Programme diversity: {unique_programmes/len(population):.2%}")
    
    # Analyze programme sparsity
    sparsities = [genome.programme.sparsity() for genome in population]
    print(f"Programme sparsity: {np.mean(sparsities):.3f} ± {np.std(sparsities):.3f}")
    
    # Analyze rule diversity
    rules = [genome.rule.table for genome in population]
    unique_rules = len(set(tuple(r) for r in rules))
    print(f"Unique rules: {unique_rules}")
    print(f"Rule diversity: {unique_rules/len(population):.2%}")
    
    # Show some example programmes
    print(f"\nExample programmes:")
    for i in range(min(5, len(population))):
        prog = population[i].programme.code
        sparsity = population[i].programme.sparsity()
        print(f"  {i+1}: {prog} (sparsity: {sparsity:.2f})")


def test_fitness_function():
    """Test if the fitness function works correctly."""
    
    print("\n🧪 FITNESS FUNCTION TEST")
    print("=" * 30)
    
    # Create fitness function with continuous=True for better gradient
    fitness = AbsoluteDifferenceFitness(continuous=True)
    
    # Test perfect outputs
    inputs = np.array([1, 2, 3, 4, 5])
    perfect_outputs = 2 * inputs
    perfect_scores = fitness(perfect_outputs, inputs)
    
    print(f"Perfect case:")
    print(f"  Inputs: {inputs}")
    print(f"  Outputs: {perfect_outputs}")
    print(f"  Scores: {perfect_scores}")
    print(f"  Mean score: {np.mean(perfect_scores):.3f}")
    
    # Test random outputs
    random_outputs = np.random.randint(0, 20, len(inputs))
    random_scores = fitness(random_outputs, inputs)
    
    print(f"\nRandom case:")
    print(f"  Inputs: {inputs}")
    print(f"  Outputs: {random_outputs}")
    print(f"  Scores: {random_scores}")
    print(f"  Mean score: {np.mean(random_scores):.3f}")
    
    # Test off-by-one outputs
    off_by_one = 2 * inputs + 1
    off_by_one_scores = fitness(off_by_one, inputs)
    
    print(f"\nOff-by-one case:")
    print(f"  Inputs: {inputs}")
    print(f"  Outputs: {off_by_one}")
    print(f"  Scores: {off_by_one_scores}")
    print(f"  Mean score: {np.mean(off_by_one_scores):.3f}")


def analyze_training_progress(trainer, input_range, targets_fn, batch_size, generations=10):
    """
    Analyze training progress step by step.
    
    Parameters
    ----------
    trainer : Trainer
        The trainer instance
    input_range : tuple
        Range of input values (min, max)
    targets_fn : callable
        Function to calculate targets from inputs (e.g., lambda x: 2*x for doubling)
    generations : int
        Number of generations to train
    """
    
    print(f"\n📈 TRAINING PROGRESS ANALYSIS")
    print("=" * 40)
    
    print(f"Training for {generations} generations...")
    
    history = {
        'best_fitness': [],
        'mean_fitness': [],
        'diversity': [],
        'accuracy': []  # Add accuracy tracking
    }
        
    # Create RNG for consistent results
    rng = np.random.default_rng(42)
    
    # Create test set for consistent accuracy measurement
    test_inputs = np.array([1, 2, 3, 4, 5, 10, 15, 20])
    test_targets = targets_fn(test_inputs)
    
    # Track best genome and its accuracy across all generations
    best_genome_ever = None
    best_accuracy_ever = 0.0
    
    for gen in range(generations):
        # Get population
        population = trainer.optimizer.ask()
        
        # Generate random inputs for this generation
        # Each genome gets a different input
        min_val, max_val = input_range
        inputs = rng.integers(min_val, max_val + 1, size=(batch_size, 1))
        
        # Calculate targets using the provided function
        targets = targets_fn(inputs)
        
        # Analyze diversity
        programmes = [genome.programme.code for genome in population]
        unique_programmes = len(set(tuple(p) for p in programmes))
        diversity = unique_programmes / len(population)
        
        # Evaluate population
        scores = trainer.evaluate_population(population, inputs, targets)
        # Normalize scores to [0, 1] range if they aren't already
        # This ensures fitness is properly scaled
        if np.max(scores) > 0:
            normalized_scores = scores / np.max(scores)
        else:
            normalized_scores = scores
        
        # Update optimizer with normalized scores
        trainer.optimizer.tell(normalized_scores)
        
        # Record stats
        best_score = np.max(scores)
        mean_score = np.mean(scores)
        
        # Calculate accuracy on test set using best genome from this generation
        best_idx = np.argmax(scores)
        best_genome = population[best_idx]
        test_outputs = trainer.evaluate_single_genome(best_genome, test_inputs)
        current_accuracy = np.mean(test_outputs == test_targets) * 100.0  # as percentage
        
        # Update best genome ever if this one is better
        if current_accuracy > best_accuracy_ever:
            best_accuracy_ever = current_accuracy
            best_genome_ever = best_genome
        
        # Use the best accuracy ever achieved for reporting
        accuracy = best_accuracy_ever
        
        history['best_fitness'].append(best_score)
        history['mean_fitness'].append(mean_score)
        history['diversity'].append(diversity)
        history['accuracy'].append(accuracy)
        
        print(f"Gen {gen:2d}: Best={best_score:.4f}, Mean={mean_score:.4f}, Diversity={diversity:.2%}, Accuracy={accuracy:.1f}%")
        
        # Analyze best genome
        if gen % 15 == 0:
            print(f"        Best programme: {best_genome.programme.code}")
            # Show test results for current best genome
            print(f"        Current test accuracy: {current_accuracy:.1f}%")
            print(f"        Best accuracy ever: {best_accuracy_ever:.1f}%")
            print(f"        Test inputs: {test_inputs}")
            print(f"        Test outputs: {test_outputs}")
            print(f"        Test targets: {test_targets}")
    
    # Return the best genome ever found along with history
    return history, best_genome_ever


def suggest_improvements():
    """Suggest improvements to make training work."""
    
    print(f"\n💡 IMPROVEMENT SUGGESTIONS")
    print("=" * 40)
    
    print("Why training isn't working:")
    print("1. 🎯 Random initialization: Rules start completely random")
    print("2. 🔍 Search space: 4^64 possible rules (huge!)")
    print("3. 🧬 Complexity: Doubling requires sophisticated rule interactions")
    print("4. 🎲 Fitness landscape: Very sparse rewards")
    
    print(f"\nSolutions to try:")
    print("1. 🌱 Better initialization:")
    print("   - Start with rules that have basic propagation")
    print("   - Use domain knowledge to seed population")
    
    print("2. 📊 Improved fitness:")
    print("   - Reward partial progress (beacon movement)")
    print("   - Use continuous fitness instead of binary")
    print("   - Multi-objective: correctness + simplicity")
    
    print("3. 🔧 Algorithm improvements:")
    print("   - Larger population size")
    print("   - Lower mutation rates")
    print("   - Elitism to preserve good solutions")
    
    print("4. 🎯 Simpler tasks first:")
    print("   - Start with identity function (f(x) = x)")
    print("   - Finally doubling (f(x) = 2x)")


def create_better_trainer(args):
    """Create a trainer with better settings for debugging."""
    
    print(f"\n🔧 CREATING IMPROVED TRAINER")
    print("=" * 40)
    
    # Setup with parameters from args
    state = StateModel([0, 1, 2, 3], immutable={0: 0})
    space = Tape1D(length=args.tape_length, radius=1)
    encoder = Em43Encoder(state, space)
    sim = Simulator(state, space, max_steps=args.max_steps, halt_thresh=args.halt_thresh)
    
    # Use continuous fitness if specified
    fitness = AbsoluteDifferenceFitness(continuous=args.continuous_fitness)
    
    # Optimizer settings from args
    optim = GAOptimizer(
        pop_size=args.pop_size,
        state=state,
        prog_len=args.prog_len,
        mutation_rate=args.mutation_rate,
        prog_mutation_rate=args.prog_mutation_rate,
        elite_fraction=args.elite_fraction,
        random_immigrant_rate=args.random_immigrant_rate
    )
    
    monitor = ConsoleLogger(log_every=2)
    trainer = Trainer(encoder, sim, fitness, optim, monitor)
    
    print("Trainer settings:")
    print(f"  Population: {optim.pop_size}")
    print(f"  Programme length: {optim.prog_len}")
    print(f"  Mutation rate: {optim.mutation_rate}")
    print(f"  Programme mutation rate: {optim.prog_mutation_rate}")
    print(f"  Elite fraction: {optim.elite_fraction}")
    print(f"  Random immigrant rate: {optim.random_immigrant_rate}")
    print(f"  Tape length: {space.length}")
    print(f"  Max steps: {sim.max_steps}")
    print(f"  Halt threshold: {sim.halt_thresh}")
    print(f"  Continuous fitness: {args.continuous_fitness}")
    
    return trainer


def main():
    """Main debugging function."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    print("🔬 Training Debugger")
    print("=" * 40)
    
    # 1. Test fitness function
    test_fitness_function()
    
    # 2. Create improved trainer with args
    trainer = create_better_trainer(args)
    
    # 3. Analyze initial population
    population = trainer.optimizer.ask()
    analyze_population_diversity(population)
    
    # 4. Run short training with random inputs each generation
    # Define input range and target function from args
    input_range = (args.input_min, args.input_max)
    target_fn = get_target_function(args.target_fn)
    
    # Train with random inputs each generation
    history, best_genome = analyze_training_progress(trainer, input_range, target_fn, args.batch_size, generations=args.generations)
    
    # 5. Plot results (unless disabled)
    if not args.no_plot:
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history['best_fitness'], 'b-', label='Best')
        plt.plot(history['mean_fitness'], 'r-', label='Mean')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(history['diversity'], 'g-')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.title('Population Diversity')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.plot(history['accuracy'], 'c-')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.scatter(history['diversity'], history['best_fitness'], alpha=0.7, label='Fitness')
        plt.scatter(history['diversity'], np.array(history['accuracy'])/100, alpha=0.7, label='Accuracy')
        plt.xlabel('Diversity')
        plt.ylabel('Performance')
        plt.title('Diversity vs Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # 6. Final evaluation of best genome
    print("\n🏆 BEST GENOME EVALUATION")
    print("=" * 40)
    print(f"Best genome programme: {best_genome.programme.code}")
    
    # Evaluate on test inputs
    test_inputs = np.array([1, 2, 3, 4, 5, 10, 15, 20])
    test_targets = target_fn(test_inputs)
    test_outputs = trainer.evaluate_single_genome(best_genome, test_inputs).squeeze()
    
    # Calculate accuracy
    accuracy = np.mean(test_outputs == test_targets) * 100.0
    
    print(f"Final test accuracy: {accuracy:.1f}%")
    print("Detailed results:")
    print("Input | Target | Output | Correct")
    print("----- | ------ | ------ | -------")
    for inp, target, output in zip(test_inputs, test_targets, test_outputs):
        correct = "✓" if output == target else "✗"
        print(f"{inp:5d} | {target:6d} | {output:6d} | {correct}")
    
    # 7. Suggestions
    suggest_improvements()
    
    print(f"\n✅ Debugging complete!")
    print("Next steps:")
    print("1. Try the improved trainer settings")
    print("2. Implement better initialization")
    print("3. Use continuous fitness")
    print("4. Start with simpler tasks")
    
    # Print best configuration for reference
    print("\nTo run with these settings again:")
    cmd = f"python {os.path.basename(__file__)}"
    cmd += f" --pop-size {args.pop_size}"
    cmd += f" --prog-len {args.prog_len}"
    cmd += f" --mutation-rate {args.mutation_rate}"
    cmd += f" --prog-mutation-rate {args.prog_mutation_rate}"
    cmd += f" --elite-fraction {args.elite_fraction}"
    cmd += f" --random-immigrant-rate {args.random_immigrant_rate}"
    cmd += f" --max-steps {args.max_steps}"
    cmd += f" --halt-thresh {args.halt_thresh}"
    cmd += f" --tape-length {args.tape_length}"
    cmd += f" --generations {args.generations}"
    cmd += f" --input-min {args.input_min}"
    cmd += f" --input-max {args.input_max}"
    cmd += f" --target-fn {args.target_fn}"
    if args.continuous_fitness:
        cmd += " --continuous-fitness"
    if args.no_plot:
        cmd += " --no-plot"
    print(cmd)


if __name__ == "__main__":
    main()
