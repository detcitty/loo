import numpy as np
import pytest
from loo.sumlogs import sumlogs

def test_python_list():
	x = [1, 2, 5, 7]
	# == np.ndarr([1,2,5,7])
	# sumlogs(x_bar, axis=None, out=int) 
	with pytest.raises(AttributeError):
		sumlogs(x)

def test_empty_list():
	x = np.array([])

	with pytest.raises(ValueError):
		sumlogs(x)

def test_int_large():
	test_values = np.array([100000, 10000000, 10000, 1000000, 100000000])

	assert sumlogs(test_values) == 100000000.0

def test_output_ndarray():
        x = np.ndarray([3, 5, 2, 5, 3])

        value = sumlogs(x, out=np.ndarray)

def test_output_list():
        x = np.ndarray([3, 3, 2, 3, 2])

        value = sumlogs(x, out=list)

def test_large_matrix():
    x = np.arange(12).reshape(3,4)

    assert sumlogs(x, axis=0) == np.array([])

def test_matrix_axis_row():
    x = np.arange(16).reshape(4,4)

    assert sumlogs(x, axis=1) == np.array([])
