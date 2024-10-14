from typing import Tuple
"""
This module provides various crossover methods for genetic algorithms.
Functions:
    single_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Perform single-point crossover on two parents to produce two children.
    uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Perform uniform crossover on two parents to produce two children.
    pooling_random_crossover(parents: Tuple[np.ndarray, ...], n_offsprings: int = 3) -> Tuple[np.ndarray, ...]:
        Perform pooling random crossover on a tuple of parents to produce a specified number of children.
"""
import numpy as np

# Define crossover methods

def single_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

def uniform_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.random.randint(0, 2, size=len(parent1))
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def pooling_random_crossover(parents: Tuple[np.ndarray,...], n_offsprings: int =3) -> Tuple[np.ndarray, ...]:
    childs = [np.zeros_like(parents[0]) for _ in range(n_offsprings)]
    for i in range(len(childs)):
        mask = np.random.randint(0, len(parents), size=parents[0].shape)
        for j in range(len(parents)):
            if j == 0:
                childs[i] = np.where(mask == j, parents[j], 0)
            else:
                childs[i] += np.where(mask == j, parents[j], 0)
    
    return childs

if __name__  == "__main__":
    parent1 = np.array([1, 2, 3, 4, 5])
    parent2 = np.array([5, 4, 3, 2, 1])
    parent3 = np.array([6, 7, 8, 9, 10])

    offsprings = pooling_random_crossover([parent1,parent2,parent3], n_offsprings=7)

    print(offsprings)