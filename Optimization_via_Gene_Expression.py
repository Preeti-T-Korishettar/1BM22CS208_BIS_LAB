#Optimization via Gene Expression Algorithms
import numpy as np
# Define the optimization function
def fitness_function(x):
    return x**2
# Convert binary string to decimal
def binary_to_decimal(binary_str):
    return int(binary_str, 2) / (2**len(binary_str) - 1) * 10 - 5  # Scale to [-5, 5]

# Initialize parameters
population_size = 20
num_genes = 10
mutation_rate = 0.1
crossover_rate = 0.7
generations = 100

# Initialize population
population = [''.join(np.random.choice(['0', '1'], num_genes)) for _ in range(population_size)]

# Main loop
for _ in range(generations):
    # Evaluate fitness
    fitness = [fitness_function(binary_to_decimal(ind)) for ind in population]

    # Selection (roulette wheel)
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    selected = np.random.choice(population, size=population_size, p=probabilities)

    # Crossover
    offspring = []
    for i in range(0, population_size, 2):
        if np.random.rand() < crossover_rate:
            point = np.random.randint(1, num_genes)
            offspring.append(selected[i][:point] + selected[i+1][point:])
            offspring.append(selected[i+1][:point] + selected[i][point:])
        else:
            offspring.append(selected[i])
            offspring.append(selected[i+1])

    # Mutation
    for i in range(population_size):
        if np.random.rand() < mutation_rate:
            point = np.random.randint(num_genes)
            offspring[i] = offspring[i][:point] + ('1' if offspring[i][point] == '0' else '0') + offspring[i][point+1:]

    population = offspring

# Output the best solution
best_individual = min(population, key=lambda ind: fitness_function(binary_to_decimal(ind)))
best_fitness = fitness_function(binary_to_decimal(best_individual))
print(“Preeti T Korishettar , 1BM22CS208”)
print(f"Best solution found: {binary_to_decimal(best_individual)}")
print(f" Fitness: {best_fitness}")
