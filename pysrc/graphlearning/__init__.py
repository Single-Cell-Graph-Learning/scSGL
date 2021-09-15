from itertools import permutations

import numpy as np
import pandas as pd

from scipy.spatial.distance import squareform

from .signedgl import learn
from ..associations import correlation, dotprod, proprho, zikendall

def learn_signed_graph(X, pos_density, neg_density, assoc="dotprod", gene_names = None, 
                       rho=10, max_iter=5000, verbose=False):

    # TODO If postive or negative density is zero, run a different algorithm
    # TODO Input check

    assocs = {"dotprod": dotprod.calc,
              "correlation": correlation.calc,
              "proprho": proprho.calc,
              "zikendall": zikendall.calc}

    if gene_names is None:
        gene_names = np.arange(1, X.shape[0]+1)

    # Check if there is any genes that has no expression at all
    nnzeros = np.count_nonzero(X, axis=1) != 0
    X_nnzeros = X[nnzeros, :]

    # Calculate association matrix
    K = assocs[assoc](X_nnzeros)
    K /= np.max(np.abs(K)) # normalize
    k = K[np.triu_indices_from(K)]
    d = np.diag(K)

    # Learn graph with desired density
    if verbose:
        print("Estimating a graph whose positive and negative edges densities are",
              "{:.3f} and {:.3f}...".format(pos_density, neg_density))

    apos_min = 0.0
    aneg_min = 0.0
    apos_max = 100.0 # TODO : Need to find upper bound
    aneg_max = 100.0 
    apos_done = False
    aneg_done = False
    densities_pos = np.zeros(50)
    densities_neg = np.zeros(50)
    for i in range(50): # Binary search
        apos = (apos_min + apos_max)/2
        aneg = (aneg_min + aneg_max)/2
    
        lpos, lneg, l = learn(k, d, apos, aneg, rho=rho, max_iter=max_iter, lpos_init="zeros", 
            lneg_init="zeros") # TODO: Clean

        densities_pos[i] = np.count_nonzero(lpos)/len(lpos)
        densities_neg[i] = np.count_nonzero(lneg)/len(lneg)

        # Check if desired density is obtained for positive part
        if not apos_done:
            if np.abs(pos_density - densities_pos[i]) < 1e-2:
                apos_done = True
            elif pos_density > densities_pos[i]:
                apos_max = apos
            elif pos_density < densities_pos[i]:
                apos_min = apos

        # Check if desired density is obtained for negative part
        if not aneg_done:
            if np.abs(neg_density - densities_neg[i]) < 1e-2:
                aneg_done = True
            elif neg_density > densities_neg[i]:
                aneg_max = aneg
            elif neg_density < densities_neg[i]:
                aneg_min = aneg

        # If desired densities are obtained, break
        if (apos_done and aneg_done):
            break

        # If binary search stuck, break
        if i>2:
            if np.abs(densities_pos[i] - densities_pos[i-1]) < 1e-6 and \
               np.abs(densities_pos[i-1] - densities_pos[i-2]) < 1e-6 and \
               np.abs(densities_neg[i] - densities_neg[i-1]) < 1e-6 and \
               np.abs(densities_neg[i-1] - densities_neg[i-2]) < 1e-6:
                break

    if verbose:
        print("Graph is found. Its positive and negative edge densities are {:.3f} and {:.3f}"\
            .format(densities_pos[i], densities_neg[i]))

    return convert_df(gene_names[nnzeros], lpos, lneg)

def convert_df(gene_names, lpos, lneg):
    gene1 = [i for i, _ in permutations(gene_names, r=2)]
    gene2 = [j for _, j in permutations(gene_names, r=2)]
    L = squareform(np.squeeze(lpos - lneg))
    edge_weights = [L[i, j] for i, j in permutations(range(len(gene_names)), r=2) if i != j]

    grn_df = pd.DataFrame({"Gene1": gene1, "Gene2": gene2, "EdgeWeight": edge_weights})
    grn_df = grn_df[grn_df.EdgeWeight != 0]
    
    return grn_df

