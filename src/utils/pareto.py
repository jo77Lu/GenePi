from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def isDominant(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Check if x dominates y
    :param x: np.ndarray
    :param y: np.ndarray
    :return: bool
    """
    # Version Faible: np.any(x < y, axis=0)
    # Version Forte: np.all(x <= y, axis=0) and np.any(x < y, axis=0) 
    return np.any(x < y, axis=0) #np.all(x <= y, axis=0) and np.any(x < y, axis=0) 

def whoIsDominant(x: Tuple[np.ndarray,...],isSorted: bool =False) -> List[int]:
    """
    Check who dominates y
    :param x: np.ndarray
    :param y: Tuple[np.ndarray,...]
    :return: Tuple[int]
    """
    if not isSorted:
        return [i for i in range(len(x)) if np.all([isDominant(x[i], y) for j, y in enumerate(x) if i != j])]
    else:
        sortedX = [np.argsort(x[:,i]) for i in range(x.shape[1])]
        print(sortedX )
        fronts=[]
        for i in range(len(x)):
            if np.all([isDominant(x[i], y) for j, y in enumerate(x) if i != j]):
                fronts.append(i)
            else:
                break
        return fronts
                


def getParetoFronts(x:Tuple[np.ndarray]) -> List[List[int]]:
    """
    Computes the Pareto fronts for a given set of points.
    Parameters:
    x (Tuple[np.ndarray]): A tuple of numpy arrays representing the points for which Pareto fronts are to be computed.
    Returns:
    List[List[int]]: A list of lists, where each sublist contains the indices of points belonging to a Pareto front.
    np.ndarray: An array where each element represents the Pareto rank of the corresponding point in the input array.
    The function iteratively identifies and removes the dominant points (Pareto front) from the set of points until no points are left.
    """

    x= np.array(x)

    paretoFrontList =[]
    paretos = np.zeros(len(x))
    index= np.array(range(len(x)))
    paretoRank=0  # Best rank is 0
    while len(x) > 0:
        listDominant = whoIsDominant(x, isSorted=False)
        if len(listDominant)==0:
            paretoFrontList.append(index)
            break

        paretoFrontList.append([index[i] for i in listDominant])
        paretos[index[listDominant]] = paretoRank

        mask = np.ones(len(x), dtype=bool)
        mask[listDominant] = False
        x = x[mask]
        index = index[mask]

        paretoRank += 1

    return paretoFrontList, paretos


def getParetoFitness(idx: int, fitness: np.ndarray, pareto_front:List[int] ) -> float:
    return 10**(pareto_front[idx]-1)+np.sum(fitness[idx])

if __name__ == '__main__':
    x = np.random.rand(200, 2)
    paretoFronts, _ = getParetoFronts(x)
    # print("Pareto Fronts:\n", paretoFronts)
    
    # Plot the vectors
    cmap = plt.colormaps.get_cmap('plasma')  # Change 'plasma' to any other colormap name
    norm = Normalize(vmin=0, vmax=len(paretoFronts) - 1)
    
    for i, front in enumerate(paretoFronts):
        front_points = np.array([x[j] for j in front])
        plt.scatter(front_points[:, 0], front_points[:, 1], color=cmap(norm(i)), label=f'Front {i+1} [{len(front)}]')
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Pareto Fronts')
    plt.legend()
    plt.show()


    # x = np.random.rand(200, 3)
    # paretoFronts, _ = getParetoFronts(x)
    # # print("Pareto Fronts:\n", paretoFronts)
    
    # # Plot the vectors
    # cmap = plt.colormaps.get_cmap('plasma')  # Change 'plasma' to any other colormap name
    # norm = Normalize(vmin=0, vmax=len(paretoFronts) - 1)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # for i, front in enumerate(paretoFronts):
    #     front_points = front
    #     print(front_points)
    #     ax.scatter(x[front_points][:, 0], x[front_points][:, 1], x[front_points][:, 2], color=cmap(norm(i)), label=f'Front {i+1} [{len(front)}]')
    
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # ax.set_title('Pareto Fronts')
    # ax.legend()
    # plt.show()