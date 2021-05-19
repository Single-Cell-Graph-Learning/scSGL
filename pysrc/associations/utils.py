import numba
import numpy as np

@numba.jit(nopython=True)
def _shuffle_columns(ct):
    # Shuffle columns of a matrix in place
    n_cols = ct.shape[1]
    for c in range(n_cols):
        np.random.shuffle(ct[:, c])

def _permutations(counts, assoc_fun, k, tau_neg, tau_pos):
    # Returns tau_neg and tau_pos percentile of the assocition values in surrogate data

    ct = counts.copy()
    thresholds = np.zeros((k, 2))
    for p in range(k):
        _shuffle_columns(ct)
        rho_rnd = assoc_fun(ct)
        thresholds[p, :] = np.percentile(rho_rnd, [tau_neg, tau_pos])

    thresholds = np.median(thresholds, axis=0)

    return thresholds