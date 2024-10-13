import numpy as np

# Define selection methods

def roulette_wheel_selection(population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
    total_fitness = np.sum(fitnesses)
    selection_probs = fitnesses / total_fitness
    parents_indices = np.random.choice(len(population), size=len(population), p=selection_probs)
    return population[parents_indices]

def tournament_selection(population: np.ndarray, fitnesses: np.ndarray, k: int = 3) -> np.ndarray:
    selected = []
    for _ in range(len(population)):
        tournament = np.random.choice(len(population), k)
        best = tournament[np.argmax(fitnesses[tournament])]
        selected.append(population[best])
    return np.array(selected)