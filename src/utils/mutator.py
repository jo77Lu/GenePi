import random
import numpy as np

# Define mutation methods

def bit_flip_mutation(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    individual[mutation_mask] = 1 - individual[mutation_mask]
    

def swap_mutation(individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            swap_idx = random.randint(0, len(individual) - 1)
            individual[i], individual[swap_idx] = individual[swap_idx], individual[i]

# Example usage
if __name__ == '__main__':
    individual = np.random.randint(0, 2, 10)
    mutation_rate = 1
    print(individual)
    bit_flip_mutation(individual[[6,2,9]], mutation_rate)
    print(individual)
    # print(mutated_individual)