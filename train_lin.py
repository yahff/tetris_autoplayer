import json
import random
import copy
import concurrent.futures

from board import Board, Shape
from adversary import RandomAdversary
from player import AIPlayer
from constants import BOARD_WIDTH, BOARD_HEIGHT, BLOCK_LIMIT

# Genetic algorithm settings.
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
        adversary = RandomAdversary(seed + i)
        
        shape_count = 0
        for event in board.run(player, adversary):
            if isinstance(event, Shape):
                shape_count += 1
                if shape_count > BLOCK_LIMIT:
                    break
        total_score += board.score
    
    return total_score / GAMES_PER_EVALUATION

def fitness_worker(lin_weights):
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
    return {
        "w1": random.uniform(-100, 0),
        "w2": random.uniform(-10, 0),
        "w3": random.uniform(1, 30),
        "w4": random.uniform(-10, 0),
        "w5": random.uniform(0.5, 5),
        "w6": random.uniform(20, 300),
        "w7": random.uniform(-5, 0),
        "w8": random.uniform(0.5, 5),
        "w9": random.uniform(-50, 0),
        "w10": random.uniform(0.1, 5),
        "w11": random.uniform(0, 100),
        "w12": random.uniform(0, 250),
        "w13": random.uniform(0, 500),
        "w14": random.uniform(50, 1000),
        "w15": random.uniform(-10, 10),
        "w16": random.uniform(-5, 5)
    }

def main():
    population = []
    for _ in range(POPULATION_SIZE):
        population.append(create_random_weights())

    best_overall_fitness = float('-inf')
    best_overall_weights = None

    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            fitnesses = list(executor.map(fitness_worker, population))
        
        for i, fitness in enumerate(fitnesses):
            print(f"  Individual {i}: Fitness = {fitness:.2f}")

        best_index = fitnesses.index(max(fitnesses))
        best_fitness = fitnesses[best_index]
        best_weights = population[best_index]
        print(f"  Best Fitness: {best_fitness:.2f}")
        print(f"  Best Weights: {best_weights}")

        if best_fitness > best_overall_fitness:
            best_overall_fitness = best_fitness
            best_overall_weights = copy.deepcopy(best_weights)
            print(f"  *** NEW OVERALL BEST! ***")
            with open('best_lin_weights.json', 'w') as f:
                json.dump(best_overall_weights, f, indent=4)

        new_population = []

        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:ELITE_COUNT]
        for i in elite_indices:
            new_population.append(population[i])

        parents = tournament_selection(population, fitnesses)
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Save the best weights
    with open('final_lin_weights.json', 'w') as f:
        json.dump(best_overall_weights, f, indent=4)
    print(f"\nTraining complete. Best fitness: {best_overall_fitness:.2f}")
    print("Best weights saved to final_lin_weights.json and best_lin_weights.json")
    print(f"Best weights: {best_overall_weights}")

if __name__ == "__main__":
    main()
