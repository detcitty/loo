from loo.psislw import psislw
import random
import numpy as np
import pytest

def test_lw_one_dimension():
    random.seed(123)
    log_n = np.random.normal(size=(50,))
    print(psislw(log_n))

def test_lw_matrix():
    np.random.seed(123)
    x = np.random.normal(size=(100,50))

    psis = psislw(x[, 1])
    lw = psis[0]


