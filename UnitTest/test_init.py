import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Test import test


def test_hello_world():
    assert test.hello_world() == "Hello World"
