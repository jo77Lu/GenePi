import random
import numpy as np

# Define mutation methods

def bit_flip_mutation(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    individual[mutation_mask] = 1 - individual[mutation_mask]
    return individual

def swap_mutation(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            swap_idx = random.randint(0, len(individual) - 1)
            individual[i], individual[swap_idx] = individual[swap_idx], individual[i]
    return individual