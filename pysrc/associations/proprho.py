import numpy as np

from . import utils

def calc(counts):
    # TODO: Docstring

    # Input validation
    if np.isnan(counts).any():
        raise ValueError("Argument 'counts' cannot have NANs.")

    if np.any(counts < 0):
        raise ValueError("Argument 'counts' cannot have negative numbers.")

    # Replace zeros in counts with minimum non-zero entry if alpha is not provided
    lr = counts.copy()
    lr[counts == 0] = np.min(counts[counts != 0])

    # Log Transformation
    lr = np.log(lr)
    ref = np.mean(lr, axis=0) # mean over samples
    lr -= ref
    
    # Calculate the proportionality metric rho
    covlr = np.cov(lr) # Covariance matrix with rows being variables
    vars = np.diag(covlr) # Variance of each row
    rho = 2*covlr/(vars + vars[..., None])
    rho[np.diag_indices_from(rho)] = 1

    return rho

def permutations(counts, k, tau_neg, tau_pos):
    # TODO: Docstring

    return utils._permutations(counts, calc, k, tau_neg, tau_pos)