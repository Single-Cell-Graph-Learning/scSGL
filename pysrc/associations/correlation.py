import numpy as np

from . import utils

def calc(counts):
    # TODO: Docstring
    
    return np.corrcoef(counts)

def permutations(counts, k, tau_neg, tau_pos):
    # TODO: Docstring

    return utils._permutations(counts, calc, k, tau_neg, tau_pos)