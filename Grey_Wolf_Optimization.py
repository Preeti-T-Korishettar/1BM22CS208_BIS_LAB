import numpy as np

# Define the fitness function (you can modify it as needed)
def fitness_function(x):
    # Example: Sphere function
    return np.sum(x**2)

# Grey Wolf Optimizer (GWO) implementation
def gwo(num_agents, dim, max_iter, fitness_func):
    # Step 1: Initialize population of grey wolves (Xi)
    X = np.random.uniform(-10, 10, (num_agents, dim))  # Random initialization of wolves' positions
    
    # Step 2: Initialize parameters a, A, and C
    a = 2  # Initial value of 'a'
    
    # Step 3: Calculate fitness of each agent
    fitness = np.apply_along_axis(fitness_func, 1, X)
    
    # Find the best three agents (Xα, Xβ, Xδ)
    sorted_indices = np.argsort(fitness)
    X_alpha = X[sorted_indices[0]]
    X_beta = X[sorted_indices[1]]
    X_delta = X[sorted_indices[2]]
    
    # Main loop (Step 4)
    for t in range(max_iter):
        # Update a
        a = 2 * (1 - t / max_iter)
        
        # Loop through all the agents
        for i in range(num_agents):
            # Step 4.1: Update A and C (Eq. 3)
            A = 2 * a * np.random.rand(dim) - a  # Eq. 4
            C = 2 * np.random.rand(dim)  # Eq. 5
            
            # Step 4.2: Update the position of the omega wolf (Eq. 6)
            D_alpha = np.abs(C * X_alpha - X[i])  # Distance from Xα
            D_beta = np.abs(C * X_beta - X[i])  # Distance from Xβ
            D_delta = np.abs(C * X_delta - X[i])  # Distance from Xδ
            
            X[i] = X[i] + A * D_alpha  # Update position based on Xα
            X[i] = X[i] + A * D_beta   # Update position based on Xβ
            X[i] = X[i] + A * D_delta  # Update position based on Xδ
        
        # Step 4.3: Calculate the fitness of all agents
        fitness = np.apply_along_axis(fitness_func, 1, X)
        
        # Step 4.4: Update Xα, Xβ, Xδ
        sorted_indices = np.argsort(fitness)
        X_alpha = X[sorted_indices[0]]
        X_beta = X[sorted_indices[1]]
        X_delta = X[sorted_indices[2]]
    
    # Step 5: Return the best solution (Xα)
    return X_alpha

# Example of usage
num_agents = 30  # Number of wolves in the population
dim = 2  # Number of dimensions
max_iter = 100  # Maximum number of iterations

# Run GWO algorithm
best_solution = gwo(num_agents, dim, max_iter, fitness_function)

print("Best solution found by GWO:", best_solution)

