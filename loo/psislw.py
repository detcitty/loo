# -*- coding: utf-8 -*-

"""Pareto smoothed importance sampling (PSIS)

This module implements Pareto smoothed importance sampling (PSIS) and PSIS
leave-one-out cross-validation for Python (Numpy).

Included functions
------------------
psislw
    Pareto smoothed importance sampling.

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
import gpdfitnew
import gpinv

def psislw(lw, wcpp=20, wtrunc=3/4, overwrite_lw=False):
    """Pareto smoothed importance sampling (PSIS).
    
    Parameters
    ----------
    lw : ndarray
        Array of size n x m containing m sets of n log weights. It is also
        possible to provide one dimensional array of length n.
    
    wcpp : number
        Percentage of samples used for GPD fit estimate (default is 20).
    
    wtrunc : float
        Positive parameter for truncating very large weights to n^wtrunc.
        Providing False or 0 disables truncation. Default values is 3/4.
    
    overwrite_lw : bool, optional
        If True, the input array `lw` is smoothed in-place. By default, a new
        array is allocated.
    
    Returns
    -------
    lw_out : ndarray
        smoothed log weights
    kss : ndarray
        Pareto tail indices
    
    """
    if lw.ndim == 2:
        n, m = lw.shape
    elif lw.ndim == 1:
        n = len(lw)
        m = 1
    else:
        raise ValueError("Argument `lw` must be 1 or 2 dimensional.")
    if n <= 1:
        raise ValueError("More than one log-weight needed.")
    
    if overwrite_lw:
        # in-place operation
        lw_out = lw
    else:
        # allocate new array for output
        lw_out = np.copy(lw, order='K')
    
    # allocate output array for kss
    kss = np.empty(m)
    
    # precalculate constants
    cutoffmin = np.log(np.finfo(float).tiny)
    logn = np.log(n)
    
    # loop over sets of log weights
    for i, x in enumerate(lw_out.T if lw_out.ndim == 2 else lw_out[None,:]):
        # improve numerical accuracy
        x -= np.max(x)
        # divide log weights into body and right tail
        xcutoff = max(
            np.percentile(x, 100 - wcpp),
            cutoffmin
        )
        expxcutoff = np.exp(xcutoff)
        tailinds, = np.where(x > xcutoff)
        x2 = x[tailinds]
        n2 = len(x2)
        if n2 <= 4:
            # not enough tail samples for gpdfitnew
            k = np.inf
        else:
            # order of tail samples
            x2si = np.argsort(x2)
            # fit generalized Pareto distribution to the right tail samples
            np.exp(x2, out=x2)
            x2 -= expxcutoff
            k, sigma = gpdfitnew(x2, sort=x2si)
            # compute ordered statistic for the fit
            sti = np.arange(0.5, n2)
            sti /= n2
            qq = gpinv(sti, k, sigma)
            qq += expxcutoff
            np.log(qq, out=qq)
            # place the smoothed tail into the output array
            x[tailinds[x2si]] = qq
        if wtrunc > 0:
            # truncate too large weights
            lwtrunc = wtrunc * logn - logn + sumlogs(x)
            x[x > lwtrunc] = lwtrunc
        # renormalize weights
        x -= sumlogs(x)
        # store tail index k
        kss[i] = k
    
    # If the provided input array is one dimensional, return kss as scalar.
    if lw_out.ndim == 1:
        kss = kss[0]
    
    return lw_out, kss