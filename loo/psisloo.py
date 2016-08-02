# -*- coding: utf-8 -*-

"""Pareto smoothed importance sampling (PSIS)

This module implements Pareto smoothed importance sampling (PSIS) and PSIS
leave-one-out cross-validation for Python (Numpy).

Included functions
------------------
psisloo
    Pareto smoothed importance sampling leave-one-out log predictive densities.

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
from loo.psislw import psislw
from loo.sumlogs import sumlogs

def psisloo(log_lik, **kwargs):
    """PSIS leave-one-out log predictive densities.
    
    Computes the log predictive densities given posterior samples of the log 
    likelihood terms p(y_i|\theta^s) in input parameter `log_lik`. Returns a 
    sum of the leave-one-out log predictive densities `loo`, individual 
    leave-one-out log predictive density terms `loos` and an estimate of Pareto 
    tail indeces `ks`. If tail index k>0.5, variance of the raw estimate does 
    not exist and if tail index k>1 the mean of the raw estimate does not exist 
    and the PSIS estimate is likely to have large variation and some bias.
    
    Parameters
    ----------
    log_lik : ndarray
        Array of size n x m containing n posterior samples of the log likelihood
        terms p(y_i|\theta^s).
    
    Additional keyword arguments are passed to the psislw() function (see the
    corresponding documentation).
    
    Returns
    -------
    loo : scalar
        sum of the leave-one-out log predictive densities
    
    loos : ndarray
        individual leave-one-out log predictive density terms
    
    ks : ndarray
        estimated Pareto tail indeces
    
    """
    # ensure overwrite flag in passed arguments
    kwargs['overwrite_lw'] = True
    # log raw weights from log_lik
    lw = -log_lik
    # compute Pareto smoothed log weights given raw log weights
    lw, ks = psislw(lw, **kwargs)
    # compute
    lw += log_lik
    loos = sumlogs(lw, axis=0)
    loo = loos.sum()
    
    return loo, loos, ks


