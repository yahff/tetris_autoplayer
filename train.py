import json
import random
import copy

from board import Board, Shape
from adversary import RandomAdversary
from player2 import AIPlayer
from constants import BOARD_WIDTH, BOARD_HEIGHT, BLOCK_LIMIT
from exceptions import BlockLimitException

# Genetic Algorithm Constants
POPULATION_SIZE = 25
NUM_GENERATIONS = 100
MUTATION_RATE = 0.2
MUTATION_STRENGTH = 0.5
TOURNAMENT_SIZE = 5
ELITE_COUNT = 3

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
        child[key] = copy.deepcopy(parent1[key])
        if isinstance(parent1[key], list):
            for i in range(len(parent1[key])):
                if isinstance(parent1[key][i], list):
                    for j in range(len(parent1[key][i])):
                        if random.random() < 0.5:
                            child[key][i][j] = parent2[key][i][j]
                else:
                    if random.random() < 0.5:
                        child[key][i] = parent2[key][i]
    return child

def mutate(weights):
    mutated_weights = copy.deepcopy(weights)
    for key in mutated_weights:
        if isinstance(mutated_weights[key], list):
            for i in range(len(mutated_weights[key])):
                if isinstance(mutated_weights[key][i], list):
                    for j in range(len(mutated_weights[key][i])):
                        if random.random() < MUTATION_RATE:
                            mutated_weights[key][i][j] += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
                else:
                    if random.random() < MUTATION_RATE:
                        mutated_weights[key][i] += random.uniform(-MUTATION_STRENGTH, MUTATION_STRENGTH)
    return mutated_weights

def main():
    # Initialize population with 8 features
    population = []
    for _ in range(POPULATION_SIZE):
        w1 = [[random.uniform(-1, 1) for _ in range(8)] for _ in range(8)]
        b1 = [random.uniform(-1, 1) for _ in range(8)]
        w2 = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(8)]
        b2 = [random.uniform(-1, 1) for _ in range(4)]
        w3 = [[random.uniform(-1, 1)] for _ in range(4)]
        b3 = [random.uniform(-1, 1)]
        population.append({'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3, 'b3': b3})

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}")

        # Evaluate fitness
        fitnesses = []
        for i, weights in enumerate(population):
            player = AIPlayer(weights=weights)
            fitness = evaluate_fitness(player)
            fitnesses.append(fitness)
            print(f"  Individual {i}: Fitness = {fitness}")

        # Find the best individual
        best_index = fitnesses.index(max(fitnesses))
        best_fitness = fitnesses[best_index]
        best_weights = population[best_index]
        print(f"  Best Fitness: {best_fitness}")

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
    with open('best_weights_player2.json', 'w') as f:
        json.dump(best_weights, f, indent=4)
    print("\nTraining complete. Best weights saved to best_weights_player2.json")

if __name__ == "__main__":
    main()
