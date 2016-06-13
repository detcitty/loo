# -*- coding: utf-8 -*-

"""Pareto smoothed importance sampling (PSIS)

This module implements Pareto smoothed importance sampling (PSIS) and PSIS
leave-one-out cross-validation for Python (Numpy).

Included functions
------------------
gpdfitnew
    Estimate the paramaters for the Generalized Pareto Distribution (GPD).

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


def gpdfitnew(x, sort=True):
    """Estimate the paramaters for the Generalized Pareto Distribution (GPD)
    
    Returns empirical Bayes estimate for the parameters of the two-parameter
    generalized Parato distribution given the data.
    
    Parameters
    ----------
    x : ndarray
        One dimensional data array
    
    sort : {bool, ndarray, 'in-place'}, optional
        If known in advance, one can provide an array of indices that would
        sort the input array `x`. If the input array is already sorted, provide
        False. If the array is not sorted but can be sorted in-place, provide
        string 'in-place'. If True (default behaviour) the sorted array indices
        are determined internally.
    
    Returns
    -------
    k, sigma : float
        estimated parameter values
    
    Notes
    -----
    This function returns a negative of Zhang and Stephens's k, because it is
    more common parameterisation.
    
    """
    if x.ndim != 1 or len(x) <= 1:
        raise ValueError("Invalid input array.")
    
    # check if x should be sorted
    if sort is True:
        sort = np.argsort(x)
        xsorted = False
    elif sort is False:
        xsorted = True
    elif sort == 'in-place':
        x.sort()
        xsorted = True
    else:
        xsorted = False
    
    n = len(x)
    m = 80 + int(np.floor(np.sqrt(n)))
    
    bs = np.arange(1, m + 1, dtype=float)
    bs -= 0.5
    np.divide(m, bs, out=bs)
    np.sqrt(bs, out=bs)
    np.subtract(1, bs, out=bs)
    if xsorted:
        bs /= 3 * x[np.floor(n/4 + 0.5) - 1]
        bs += 1 / x[-1]
    else:
        bs /= 3 * x[sort[np.floor(n/4 + 0.5) - 1]]
        bs += 1 / x[sort[-1]]
    
    ks = np.negative(bs)
    temp = ks[:,None] * x
    np.log1p(temp, out=temp)
    np.mean(temp, axis=1, out=ks)
    
    L = bs / ks
    np.negative(L, out=L)
    np.log(L, out=L)
    L -= ks
    L -= 1
    L *= n
    
    temp = L - L[:,None]
    np.exp(temp, out=temp)
    w = np.sum(temp, axis=1)
    np.divide(1, w, out=w)
    
    # remove negligible weights
    dii = w >= 10 * np.finfo(float).eps
    if not np.all(dii):
        w = w[dii]
        bs = bs[dii]
    # normalise w
    w /= w.sum()
    
    # posterior mean for b
    b = np.sum(bs * w)
    # Estimate for k, note that we return a negative of Zhang and
    # Stephens's k, because it is more common parameterisation.
    temp = (-b) * x
    np.log1p(temp, out=temp)
    k = np.mean(temp)
    # estimate for sigma
    sigma = -k / b
    
    return k, sigma