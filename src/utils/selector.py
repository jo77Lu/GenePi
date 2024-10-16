from typing import Type
import numpy as np
import pareto
from abc import ABC, abstractmethod

class SelectionFunction(ABC):
    @abstractmethod
    def select(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        pass

class RouletteWheelSelection(SelectionFunction):
    def __init__(self, k: int = 3):
        self.k = k

    def select(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        total_fitness = np.sum(fitnesses)
        selection_probs = fitnesses / total_fitness
        parents_indices = np.random.choice(len(population), size=len(population), p=selection_probs)
        return population[parents_indices]

class TournamentSelection(SelectionFunction):
    def __init__(self, k: int = 3):
        self.k = k

    def select(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        selected = []
        for _ in range(len(population)):
            tournament = np.random.choice(len(population), self.k)
            best = tournament[np.argmax(fitnesses[tournament])]
            selected.append(population[best])
        return np.array(selected)

class ParetoSelection(SelectionFunction):
    def __init__(self, fun_selection: Type[SelectionFunction] = TournamentSelection):
        self.fun_selection = fun_selection

    def select(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        pareto_fronts = pareto.getParetoFronts(fitnesses)
        selected = []
        for front in pareto_fronts:
            selected.append(front[self.fun_selection.select(population[front], fitnesses[front])])
        return np.concatenate(selected)

# Example usage
if __name__ == '__main__':
    population = np.random.rand(100, 10)
    fitnesses = np.random.rand(100, 2)
    selection_function = ParetoSelection()
    selected_population = selection_function.select(population, fitnesses)
    print(selected_population)