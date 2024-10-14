import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the parent folder and all its subdirectories to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, parent_dir)
utils_dir = os.path.join(parent_dir, 'utils')
sys.path.insert(0, utils_dir)

from Test_to_be_deleted import test
from utils.CrossOver import combinator
from utils.Selection import selector
from utils.Mutation import mutator


def test_hello_world():
    assert test.hello_world() == "Hello World"
