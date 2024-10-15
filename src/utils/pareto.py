from typing import Tuple, Generator, List
import numpy as np


def isDominant(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Check if x dominates y
    :param x: np.ndarray
    :param y: np.ndarray
    :return: bool
    """
    return np.all(x <= y, axis=0) and np.any(x < y, axis=0)

def whoIsDominant(x: Tuple[np.ndarray,...]) -> List[int]:
    """
    Check who dominates y
    :param x: np.ndarray
    :param y: Tuple[np.ndarray,...]
    :return: Tuple[int]
    """
    return [i for i in range(len(x)) if np.all([isDominant(x[i], y) for j, y in enumerate(x) if i != j])]


def getParetoFronts(x:Tuple[np.ndarray]) -> List[List[int]]:

    paretoFrontList =[]
    index= [i for i in range(len(x))]

    while len(x) > 0:
        listDominant = whoIsDominant(x)
        if len(listDominant)==0:
            paretoFrontList.append(x)
            break

        paretoFrontList.append([index[i] for i in listDominant])
        x = np.delete(x, listDominant, axis=0)
        index = np.delete(index, listDominant, axis=0)
    return paretoFrontList