import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
# Import Modules
from GenePi.src.utils import combinator
from GenePi.src.utils import selector
from GenePi.src.utils import mutator

class GeneticAlgorithm:
    def __init__(self, 
                 pop_size: int, 
                 gene_length: int, 
                 generations: int, 
                 mutation_rate: float, 
                 crossover_func: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], 
                 mutation_func: Callable[[np.ndarray, float], np.ndarray], 
                 selection_func: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        self.pop_size = pop_size
        self.gene_length = gene_length
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.selection_func = selection_func
        self.population = self.initialize_population()
        self.best_fitnesses = []

    def initialize_population(self) -> np.ndarray:
        return np.random.randint(2, size=(self.pop_size, self.gene_length))

    def fitness(self, individual: np.ndarray) -> int:
        return np.sum(individual)

    def plot_fitness(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.generations)
        self.ax.set_ylim(0, self.gene_length)
        self.line, = self.ax.plot([], [], 'r-')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')

    def update_plot(self):
        self.line.set_xdata(range(len(self.best_fitnesses)))
        self.line.set_ydata(self.best_fitnesses)
        self.ax.set_xlim(0, self.generations)
        self.ax.set_ylim(0, max(self.best_fitnesses) + 1)
        plt.draw()
        plt.pause(0.01)

    def evolve(self) -> np.ndarray:
        self.plot_fitness()

        for _ in range(self.generations):
            fitnesses = np.array([self.fitness(ind) for ind in self.population])
            parents = self.selection_func(self.population, fitnesses)
            
            next_generation = []
            for i in range(0, len(parents), 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = self.crossover_func(parent1, parent2)
                next_generation.extend([self.mutation_func(child1, self.mutation_rate), self.mutation_func(child2, self.mutation_rate)])
            
            self.population = np.array(next_generation)

            # Track the best fitness
            best_fitness = max(fitnesses)
            self.best_fitnesses.append(best_fitness)

            # Update the plot
            self.update_plot()

        plt.ioff()
        plt.show()

        # Return the best individual
        best_individual = max(self.population, key=self.fitness)
        return best_individual
    

if __name__ == "__main__":
    # Parameters
    pop_size = 10
    gene_length = 8
    generations = 20
    mutation_rate = 0.01

    # Run the genetic algorithm with different strategies
    ga = GeneticAlgorithm(
        pop_size, gene_length, generations, mutation_rate,
        crossover_func=combinator.single_point_crossover,
        mutation_func=mutator.bit_flip_mutation,
        selection_func=selector.roulette_wheel_selection
    )
    best_solution = ga.evolve()
    print("Best solution:", best_solution)
    print("Best solution fitness:", ga.fitness(best_solution))

