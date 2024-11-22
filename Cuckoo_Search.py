import numpy as np

# Objective function to optimize (you can replace this with your own function)
def objective_function(x):
    return np.sum(x**2)  # Simple example: minimizing the sum of squares

# Generate an initial population of n host nests Xi (i = 1, 2, ..., n)
def generate_initial_population(n, dim, bounds):
    return np.random.uniform(bounds[0], bounds[1], (n, dim))

# Simplified Levy flight for cuckoo's movement (no external libraries)
def levy_flight(dim, beta=1.5):
    # Simulating Levy flight using a Cauchy distribution (similar behavior to Levy flight)
    step = np.random.standard_cauchy(size=dim)  # Cauchy distribution as an approximation
    return step

# Cuckoo Search Algorithm Implementation
def cuckoo_search(objective_function, n, dim, bounds, max_iter, Pa=0.25):
    # Step 1: Generate the initial population
    nests = generate_initial_population(n, dim, bounds)
    fitness = np.array([objective_function(x) for x in nests])

    # Best solution found so far
    best_nest = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    t = 0  # Initialize iteration counter

    while t < max_iter:
        # Step 2: Generate a cuckoo randomly by Levy flights
        cuckoo = best_nest + levy_flight(dim)  # This is a random cuckoo position
        
        # Bound the cuckoo solution within the defined search space
        cuckoo = np.clip(cuckoo, bounds[0], bounds[1])
        
        # Step 3: Evaluate the cuckoo's fitness
        cuckoo_fitness = objective_function(cuckoo)

        # Step 4: Choose a nest randomly and compare the fitness
        j = np.random.randint(n)
        
        if cuckoo_fitness > fitness[j]:
            nests[j] = cuckoo
            fitness[j] = cuckoo_fitness

        # Step 5: Abandon a fraction (Pa) of worse nests
        # Sort nests based on fitness values (ascending)
        sorted_indices = np.argsort(fitness)
        nests = nests[sorted_indices]
        fitness = fitness[sorted_indices]
        
        # Abandon worst Pa fraction of nests and create new ones
        num_to_abandon = int(Pa * n)
        nests[-num_to_abandon:] = generate_initial_population(num_to_abandon, dim, bounds)
        fitness[-num_to_abandon:] = np.array([objective_function(x) for x in nests[-num_to_abandon:]])

        # Keep the best solution
        best_nest = nests[0]
        best_fitness = fitness[0]

        t += 1

    return best_nest, best_fitness

# Example Usage
n = 50  # Population size
dim = 2  # Dimension of the solution space
bounds = (-5, 5)  # Search space bounds for each dimension
max_iter = 100  # Maximum number of iterations

# Run the Cuckoo Search algorithm
best_solution, best_value = cuckoo_search(objective_function, n, dim, bounds, max_iter)

print("Best solution found: ", best_solution)
print("Best fitness value: ", best_value)



