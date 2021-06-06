from pysrc import associations
import numba
import numpy as np

from ..utils import progress_bar

@numba.jit(nopython=True)
def _shuffle_columns(ct):
    # Shuffle columns of a matrix in place
    n_cols = ct.shape[1]
    for c in range(n_cols):
        np.random.shuffle(ct[:, c])

def _permutations(counts, assoc_fun, k, tau_neg, tau_pos):
    # Returns tau_neg and tau_pos percentile of the assocition values in surrogate data

    ct = counts.copy()
    thresholds = np.zeros((len(tau_neg), k, 2))
    for p in range(k):
        _shuffle_columns(ct)
        rho_rnd = assoc_fun(ct)
        rho_rnd = rho_rnd[np.triu_indices_from(rho_rnd, k=1)]
        
        for i in range(len(tau_neg)):
            thresholds[i, p, :] = np.percentile(rho_rnd, [tau_neg[i], tau_pos[i]])
        
        progress_bar.show(p+1, k, prefix="Estimating graph density")

    thresholds = np.median(thresholds, axis=1)

    return thresholds

def _associations(counts, assoc_fun, k):
    # Returns tau_neg and tau_pos percentile of the assocition values in surrogate data

    n_genes, _ = counts.shape
    associations = np.zeros((k, n_genes*(n_genes-1)//2))
    for p in range(k):
        ct = counts.copy()
        _shuffle_columns(ct)
        rho_rnd = assoc_fun(ct)
        associations[p, :] = rho_rnd[np.triu_indices_from(rho_rnd, k=1)]
        
        progress_bar.show(p+1, k, prefix="Calculating associations in surrogate data: ")

    return associations