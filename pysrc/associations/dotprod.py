import numpy as np

from . import utils

def calc(counts):
    # TODO: Docstring
    K = counts@counts.T
    return K

def permutations(counts, k, tau_neg, tau_pos):
    # TODO: Docstring

    return utils._permutations(counts, calc, k, tau_neg, tau_pos)

def associations(counts, k):
    # TODO: Docstring
    
    return utils._associations(counts, calc, k)