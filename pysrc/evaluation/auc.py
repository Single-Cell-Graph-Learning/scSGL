from itertools import product, combinations, permutations

import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score

def unsigned(true_edges, pred_edges, type):

    unique_nodes = np.unique(true_edges.loc[:,['Gene1','Gene2']])
    if type == "undirected":
        possible_edges = list(combinations(unique_nodes, r = 2))
    elif type == "directed":
        possible_edges = list(permutations(unique_nodes, r = 2))
    elif type == "tfedges":
        possible_edges = set(product(set(true_edges.Gene1), set(unique_nodes)))

    true_edges_dict = {'|'.join(p):0 for p in possible_edges}
    pred_edges_dict = {'|'.join(p):0 for p in possible_edges}

    for edge in true_edges.itertuples():
        # Ignore self-edges
        if edge.Gene1 == edge.Gene2:
            continue

        if "|".join((edge.Gene1, edge.Gene2)) in true_edges_dict:
            true_edges_dict["|".join((edge.Gene1, edge.Gene2))] = 1
        
        if type == "undirected":
            if "|".join((edge.Gene2, edge.Gene1)) in true_edges_dict:
                true_edges_dict["|".join((edge.Gene2, edge.Gene1))] = 1

    for edge in pred_edges.itertuples():
        # Ignore self-edges
        if edge.Gene1 == edge.Gene2:
            continue

        if "|".join((edge.Gene1, edge.Gene2)) in pred_edges_dict:
            if np.abs(edge.EdgeWeight) > pred_edges_dict["|".join((edge.Gene1, edge.Gene2))]:
                pred_edges_dict["|".join((edge.Gene1, edge.Gene2))] = np.abs(edge.EdgeWeight)

    auprc = average_precision_score(list(true_edges_dict.values()), 
                                    list(pred_edges_dict.values()))

    auroc = roc_auc_score(list(true_edges_dict.values()), 
                          list(pred_edges_dict.values()))

    random_auprc = np.sum(np.array(list(true_edges_dict.values())))/len(true_edges_dict)
    auprc_ratio = auprc/random_auprc
    auroc_ratio = auroc/0.5

    return auprc, auroc, auprc_ratio, auroc_ratio
