from loo.psislw import psislw
import random
import numpy as np
import pytest

def test_lw_one_dimension():

  random.seed(123)
  log_n = np.random.normal(size=(50,))
  print(psislw(log_n))

test_lw_one_dimension()