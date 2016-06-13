# -*- coding: utf-8 -*-

"""Pareto smoothed importance sampling (PSIS)

This module implements Pareto smoothed importance sampling (PSIS) and PSIS
leave-one-out cross-validation for Python (Numpy).

Included functions
------------------
sumlogs
    Sum of vector where numbers are represented by their logarithms.

References
----------
Aki Vehtari, Andrew Gelman and Jonah Gabry (2015). Efficient implementation
of leave-one-out cross-validation and WAIC for evaluating fitted Bayesian
models. arXiv preprint arXiv:1507.04544.

Aki Vehtari and Andrew Gelman (2015). Pareto smoothed importance sampling.
arXiv preprint arXiv:1507.02646.

"""

# Copyright (c) 2015 Aki Vehtari, Tuomas Sivula
# Original Matlab version by Aki Vehtari. Translation to Python
# by Tuomas Sivula.

# This software is distributed under the GNU General Public
# License (version 3 or later); please refer to the file
# License.txt, included with the software, for details.

from __future__ import division # For Python 2 compatibility
import numpy as np


def sumlogs(x, axis=None, out=None):
    """Sum of vector where numbers are represented by their logarithms.
    
    Calculates np.log(np.sum(np.exp(x), axis=axis)) in such a fashion that it
    works even when elements have large magnitude.
    
    """
    maxx = x.max(axis=axis, keepdims=True)
    xnorm = x - maxx
    np.exp(xnorm, out=xnorm)
    out = np.sum(xnorm, axis=axis, out=out)
    if isinstance(out, np.ndarray):
        np.log(out, out=out)
    else:
        out = np.log(out)
    out += np.squeeze(maxx)
    return out

