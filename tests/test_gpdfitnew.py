"""

library(loo)

context("gpdfit")
test_that("gpdfit returns correct result", {
  set.seed(123)
  x <- rexp(100)
  gpdfit_val <- unlist(gpdfit(x))
  gpdfit_ans <- structure(c(0.0274030348712631, 1.01829821712701),
                          .Names = c("k", "sigma"))
  expect_equal(gpdfit_val, gpdfit_ans)
})

"""

import numpy as np
import pytest
from loo.gpdfitnew import gpdfitnew


def test_emptyArray():
	x = np.ndarray([])
	with pytest.raises(ValueError):
		gpdfitnew(x)

def test_sort():
	x = np.ones((5, ), dtype=bool)
	print x
	assert gpdfitnew(x) 

def test_unsorted():
	x = np.ones((100,100), dtype=bool)
	print x

