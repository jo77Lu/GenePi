import sys  # noqa: E402
import os  # noqa: E402
import pytest  # noqa: E402
import numpy as np  # noqa: E402

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # noqa: E402

# Add the parent folder and all its subdirectories to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))  # noqa: E402
sys.path.insert(0, parent_dir)  # noqa: E402
utils_dir = os.path.join(parent_dir, 'utils')  # noqa: E402
sys.path.insert(0, utils_dir)  # noqa: E402

from Test_to_be_deleted import test  # noqa: E402
from utils import combinator  # noqa: E402
from utils import selector  # noqa: E402
from utils import mutator  # noqa: E402
from utils import pareto  # noqa: E402
# noqa: E402

def test_hello_world():
    assert test.hello_world() == "Hello World"


# TEST COMBINATOR
@pytest.fixture
def parents():
    parent1 = np.array([1, 2, 3, 4, 5])
    parent2 = np.array([6, 7, 8, 9, 10])
    parent3 = np.array([11, 12, 13, 14, 15])
    return (parent1, parent2, parent3)

@pytest.fixture
def scores():
    score1 = np.array([1, 2, 3,4])
    score2 = np.array([6, 7, 8,9])
    score3 = np.array([11, 12, 13,14])
    return (score1, score2, score3)


def test_pooling_random_crossover(parents):
    offsprings = combinator.pooling_random_crossover(parents, n_offsprings=3)
    assert len(offsprings) == 3, f"Expected 3 offsprings, got {len(offsprings)}"
    for child in offsprings:
        assert len(child) == len(parents[0]), f"Expected child length {len(parents[0])}, got {len(child)}"
        assert np.all(np.logical_or.reduce([np.isin(child, parent) for parent in parents])), "Child contains elements not found in either parent"

def test_isDominant(scores):
    assert pareto.isDominant(scores[0],scores[1]), "Expected True, got False"
    assert not pareto.isDominant(scores[2],scores[1]), "Expected False, got True"
    assert not pareto.isDominant(np.array([1,1]),np.array([0.5,0.5])), "Testing [1,0.5],[0.5,1] : Expected False, got True"
    assert pareto.isDominant(np.array([0.5,0.5]),np.array([1,1])), " testing [0.5,0.5],[1,1] : Expected True, got False"

def test_whoIsDominant(scores):
    assert pareto.whoIsDominant(scores) == [0], f"Expected [0], got {pareto.whoIsDominant(scores)}"
    test2 = (np.array([1,0]),np.array([0,1]),np.array([0.5,0.5]))
    assert pareto.whoIsDominant(test2) == [0,1,2], f"Expected [0,1,2], got {pareto.whoIsDominant(test2)}"


def test_getParetoFronts(scores):
    assert pareto.getParetoFronts(scores) == [[0],[1],[2]], f"Expected [[0],[1],[2]], got {pareto.getParetoFronts(scores)}"

def test_roulette_wheel_selection():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    #Fix the seed for reproducibility
    np.random.seed(0)

    #Generate random population and fitnesses
    population = np.random.rand(100, 10)
    fitnesses = np.random.rand(100)

    #Test the tournament selection
    selection_function = selector.RouletteWheelSelection()
    selected_population = selection_function.select(population, fitnesses)

    # 1 test shape
    assert selected_population.shape == population.shape, f"Expected shape {population.shape}, got {selected_population.shape}"

    # 2 test values
    if os.path.exists(os.path.join(current_dir,"refData/test_roulette_wheel_selection.npy")):
        np.testing.assert_allclose(selected_population, np.load(os.path.join(current_dir,"refData/test_roulette_wheel_selection.npy")), rtol=1e-5, atol=1e-8)
    else:
        np.save(os.path.join(current_dir,"refData/test_roulette_wheel_selection.npy"), selected_population)
        

def test_tournament_selection():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    #Fix the seed for reproducibility
    np.random.seed(0)

    #Generate random population and fitnesses
    population = np.random.rand(100, 10)
    fitnesses = np.random.rand(100)

    #Test the tournament selection
    selection_function = selector.TournamentSelection()
    selected_population = selection_function.select(population, fitnesses)

    # 1 test shape
    assert selected_population.shape == population.shape, f"Expected shape {population.shape}, got {selected_population.shape}"

    # 2 test values
    if os.path.exists(os.path.join(current_dir,"refData/test_tournament_selection.npy")):
        np.testing.assert_allclose(selected_population, np.load(os.path.join(current_dir,"refData/test_tournament_selection.npy")), rtol=1e-5, atol=1e-8)
    else:
        np.save(os.path.join(current_dir,"refData/test_tournament_selection.npy"), selected_population)