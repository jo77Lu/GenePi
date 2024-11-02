from typing import Type, List, Optional, Callable
import numpy as np
import pareto
from abc import ABC, abstractmethod

class SelectionFunction(ABC):
    @abstractmethod
    def select(self, fitnesses: np.ndarray) -> List[int]:
        pass

class RouletteWheelSelection(SelectionFunction):
    def __init__(self, k: int = 3):
        self.k = k

    def select(self, fitnesses: np.ndarray) -> List[int]:
        """
        Selects k individuals from the population using roulette wheel selection
            :param population: np.ndarray
            :param fitnesses: np.ndarray
            :return: np.ndarray
        """
        total_fitness = np.sum(fitnesses)
        selection_probs = fitnesses / total_fitness

        return np.random.choice(len(fitnesses), size=len(fitnesses), p=selection_probs, replace=False)



class TournamentSelection(SelectionFunction):
    def __init__(self, k: int = 3):
        self.k = k

    def select(self, fitnesses: np.ndarray, key_function: Optional[Callable[[int], float]] = None) -> List[int]:
        """
        Selects individuals from the population using tournament selection
            :param population: np.ndarray
            :param fitnesses: np.ndarray
            :return: np.ndarray of selected individual indices
        """
        if len(fitnesses) < self.k:
            idx_tournament  = np.arange(len(fitnesses)-1)
        else:
            idx_tournament = np.random.choice(len(fitnesses), self.k,replace=False)

        if key_function:
            sorted_indices = sorted(idx_tournament, key=lambda x: key_function(x))
        else:
            sorted_indices = sorted(idx_tournament, key=lambda x: fitnesses[x])

        idx_selected = sorted_indices[:self.k]

        return idx_selected

class ParetoTournamentSelection(SelectionFunction):
    def __init__(self, fun_selection: Type[SelectionFunction] = TournamentSelection):
        self.fun_selection = fun_selection()

    def select(self, fitnesses: np.ndarray) -> List[int]:
        """
        Selects individuals from the population using tournament selection
            :param population: np.ndarray
            :param fitnesses: np.ndarray
            :return: np.ndarray of selected individual indices
        """
        if len(population) <= self.k:
            # idx_tournament = np.arange(len(population)) 
            idx_selected  = np.arange(len(fitnesses))
        else:
            idx_tournament = np.random.choice(len(fitnesses), self.k,replace=False)

            idx_Pareto_sorted = np.argsort(pareto[idx_tournament])
            idx_selected = idx_tournament[np.argmin(fitnesses[idx_tournament])]

        return idx_selected

# Example usage
if __name__ == '__main__':
    population = np.random.rand(100, 10)
    fitnesses = np.random.rand(100, 2)
    # selection_function = ParetoTournamentSelection()
    # selected_population = selection_function.select(population, fitnesses, pareto.getParetoFronts(fitnesses))
    # print(selected_population)

    selection_function = TournamentSelection()
    _ , paretos=pareto.getParetoFronts(fitnesses)
    selected_population = selection_function.select(fitnesses, key_function=lambda x: pareto.getParetoFitness(x, fitnesses,paretos ))
    print(selected_population)