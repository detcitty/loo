import pytest
import random
import numpy as np
from loo.psisloo import psisloo

def test_loo():
  random.seed(123)
  x = np.random.normal(size=(100, 50))
  ll = loo.psisloo(x)
  print(ll)