#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.stats

def create_gaussian_curves(mu=0, sd=1, N=1000, xmin=-10, xmax=10, nbx=300, sd_noise=0):
    """
    Create Gaussian curves. Discretizes the curves in the range [xmin,xmax]
    Parameters:
        mu: a number or a vector of length N with the different values of $\\mu$
        sd: a number or a vector of length N with the different values of $\\sigma$
        N: number of curves
        xmin: minimal value for x
        xmax: maximal value for x
        nbx: number of points in the interval $[xmin, xmax]$
        sd_noise: a positive number with the standard deviation of the noise
    Returns:
        x: array of shape (nbx,) - x coordinates
        curves: array of shape (N, nbx) - Gaussian curves with optional noise
    """
     # Handle mu parameter 
    mu = np.asarray(mu)
    if mu.ndim == 0 or mu.shape == ():  # Scalar
        mu = mu * np.ones(N)
    mu = mu.reshape(-1, 1)  # Reshape to (N, 1)    
    # Handle sd parameter
    sd = np.asarray(sd)
    if sd.ndim == 0 or sd.shape == ():  # Scalar
        sd = sd * np.ones(N)
    sd = sd.reshape(-1, 1)  # Reshape to (N, 1) 
    # Verify dimensions
    assert mu.shape[0] == N, f"mu has shape {mu.shape}, expected ({N},) or scalar"
    assert sd.shape[0] == N, f"sd has shape {sd.shape}, expected ({N},) or scalar"

    # Create x values
    x = np.linspace(xmin, xmax, nbx)
    # Create Gaussian distributions for each (mu, sd) pair
    # Note: scipy.stats.norm can handle vectorized parameters
    normal = scipy.stats.norm(loc=mu, scale=sd)
    # Compute PDF values and add noise
    curves = normal.pdf(x)  # Shape: (N, nbx)
    
    return x, curves + sd_noise * np.random.normal(0., 1., size=(N, nbx))