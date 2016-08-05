import pytest
import random
import numpy as np
from loo.psisloo import psisloo

def test_loo():
    random.seed(123)
    x = np.random.normal(size=(100, 50))
    ll = psisloo(x)
    print(ll)

    loo_ans = np.array([-24.2339828660691, 48.6639600634871, 48.4679657321382, 
                        0.706772789390649, 0.990389359906155, 1.4135455787813])

    
def test_func_matrix_equal():
    pass

def test_expect_errors():
    pass


