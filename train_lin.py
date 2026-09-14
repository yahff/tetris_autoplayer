import json
import random
import copy
import concurrent.futures

from board import Board, Shape
from adversary import RandomAdversary
from player import AIPlayer
from constants import BOARD_WIDTH, BOARD_HEIGHT, BLOCK_LIMIT
from exceptions import BlockLimitException

# Genetic Algorithm Constants
POPULATION_SIZE = 50
NUM_GENERATIONS = 150
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.4
TOURNAMENT_SIZE = 6
ELITE_COUNT = 5

GAMES_PER_EVALUATION = 3

def evaluate_fitness(player, seed=None):
    if seed is None:
        seed = random.randint(0, 100000)
    
    total_score = 0
    for i in range(GAMES_PER_EVALUATION):
        board = Board(BOARD_WIDTH, BOARD_HEIGHT)
        adversary = RandomAdversary(seed + i)  # Different seed for each game
        
        # Count placed blocks via Shape events and break when limit reached.
        # Shapes are yielded once at start and once per placed piece.
        shape_count = 0
        for event in board.run(player, adversary):
            if isinstance(event, Shape):
                shape_count += 1
                # placed = shape_count - 1; stop when placed hits BLOCK_LIMIT
                if shape_count > BLOCK_LIMIT:
                    break
        total_score += board.score
    
    return total_score / GAMES_PER_EVALUATION

def fitness_worker(lin_weights):
    """Worker function for parallel fitness evaluation"""
    player = AIPlayer(lin_weights=lin_weights)
    return evaluate_fitness(player)

def tournament_selection(population, fitnesses):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
        winner = max(tournament, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(weights):
    mutated_weights = copy.deepcopy(weights)
    for key in mutated_weights:
        if random.random() < MUTATION_RATE:
            mutated_weights[key] += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
    return mutated_weights

def create_random_weights():
    """Create a random set of 16 linear weights with EXTREME diversity"""
    return {
        "w1": random.uniform(-100, 0),         # holes penalty
        "w2": random.uniform(-10, 0),          # bumpiness penalty
        "w3": random.uniform(1, 30),           # max_height threshold
        "w4": random.uniform(-10, 0),          # max_height penalty coefficient
        "w5": random.uniform(0.5, 5),          # max_height penalty exponent
        "w6": random.uniform(20, 300),         # aggregate_height threshold
        "w7": random.uniform(-5, 0),           # aggregate_height penalty coefficient
        "w8": random.uniform(0.5, 5),          # aggregate_height penalty exponent
        "w9": random.uniform(-50, 0),          # potential clears penalty coefficient
        "w10": random.uniform(0.1, 5),         # potential clears penalty exponent
        "w11": random.uniform(0, 100),         # 1 line cleared reward
        "w12": random.uniform(0, 250),         # 2 lines cleared reward
        "w13": random.uniform(0, 500),         # 3 lines cleared reward
        "w14": random.uniform(50, 1000),       # 4+ lines cleared reward
        "w15": random.uniform(-10, 10),        # max_height growth promotion
        "w16": random.uniform(-5, 5)           # aggregate_height growth promotion
    }

def main():
    # Initialize population
    population = []
    for _ in range(POPULATION_SIZE):
        population.append(create_random_weights())

    best_overall_fitness = float('-inf')
    best_overall_weights = None

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}")

        # Evaluate fitness in parallel across all CPU cores
        with concurrent.futures.ProcessPoolExecutor() as executor:
            fitnesses = list(executor.map(fitness_worker, population))
        
        for i, fitness in enumerate(fitnesses):
            print(f"  Individual {i}: Fitness = {fitness:.2f}")

        # Find the best individual
        best_index = fitnesses.index(max(fitnesses))
        best_fitness = fitnesses[best_index]
        best_weights = population[best_index]
        print(f"  Best Fitness: {best_fitness:.2f}")
        print(f"  Best Weights: {best_weights}")

        # Track overall best
        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_weights = copy.deepcopy(best_weights)
            print(f"  *** NEW OVERALL BEST! ***")
            # Save intermediate best weights
            with open('best_lin_weights1.json', 'w') as f:
                json.dump(best_overall_weights, f, indent=4)

        # Create the next generation
        new_population = []

        # Elitism
        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:ELITE_COUNT]
        for i in elite_indices:
            new_population.append(population[i])

        # Selection, Crossover, and Mutation
        parents = tournament_selection(population, fitnesses)
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Save the best weights
    with open('final_lin_weights1.json', 'w') as f:
        json.dump(best_overall_weights, f, indent=4)
    print(f"\nTraining complete. Best fitness: {best_overall_fitness:.2f}")
    print(f"Best weights saved to final_lin_weights1.json and best_lin_weights1.json")
    print(f"Best weights: {best_overall_weights}")

if __name__ == "__main__":
    main()
