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
from utils.CrossOver import combinator  # noqa: E402
from utils.Selection import selector  # noqa: E402
from utils.Mutation import mutator  # noqa: E402


def test_hello_world():
    assert test.hello_world() == "Hello World"


# TEST COMBINATOR
@pytest.fixture
def parents():
    parent1 = np.array([1, 2, 3, 4, 5])
    parent2 = np.array([6, 7, 8, 9, 10])
    parent3 = np.array([11, 12, 13, 14, 15])
    return (parent1, parent2, parent3)


def test_pooling_random_crossover(parents):
    offsprings = combinator.pooling_random_crossover(parents, n_offsprings=3)
    assert len(offsprings) == 3, f"Expected 3 offsprings, got {len(offsprings)}"
    for child in offsprings:
        assert len(child) == len(parents[0]), f"Expected child length {len(parents[0])}, got {len(child)}"
        assert np.all(np.logical_or.reduce([np.isin(child, parent) for parent in parents])), "Child contains elements not found in either parent"