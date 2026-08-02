import numpy as np
import networkx as nx
import random
import math
from itertools import permutations
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from typing import Tuple, Optional


def new_society(N, B, L, T, alpha):
    ''' 

    '''
    # --- init base layer and attrs ---
    G = nx.DiGraph()
    G.add_nodes_from(B, level=1)
    non_basal = list(range(N-B))

    # --- begin addition process ---

    # create list once and append to it for faster runtime
    available_nodes = list(G.nodes())
    for n in non_basal:
        subordinate = random.choice(available_nodes)
        G.add_edge(n, subordinate)
        G.nodes[n]['level'] = G.nodes[subordinate]['level'] + 1
        available_nodes.append(n)

    del available_nodes  # no longer needed

    # --- add excess links ---

    levels = np.array([G.nodes[n]['level'] for n in range(N)])
    sources = np.arange(N)
    targets = np.arange(B, N)  # non-basal nodes

    # 2d grid of all valid pairs
    i_grid, j_grid = np.meshgrid(sources, targets, indexing='ij')

    # broadcast levels to calculate all level differences at once
    s_i = levels[sources][:, np.newaxis]
    s_j = levels[targets][np.newaxis, :]
    level_diff = s_i - s_j

    # vectorized weight calculation
    raw_weights = np.exp(-((level_diff - 1) ** 2) / (2 * T ** 2))

    # mask out edges that already exist in the tree to prevent duplicates
    existing_edges_mask = np.zeros((N, N - B), dtype=bool)
    for u, v in G.edges():
        if v >= B:  # only care if target is non-basal
            existing_edges_mask[u, v - B] = True
    raw_weights[existing_edges_mask] = 0.0
    raw_weights[np.arange(N-B), np.arange(N-B)] = 0.0  # self-loops mask

    # flatten arrays to prep for probability sampling
    flat_pairs_idx = np.arange(N * (N - B))
    flat_weights = raw_weights.ravel()

    # total sum of valid weights
    weight_sum = np.sum(flat_weights)
    if weight_sum == 0:
        print("Error: Weights not added.")
        return G

    # normalize
    probabilities = flat_weights / weight_sum

    # sample all excess links simultaneously without replacement
    num_excess_links = L - G.number_of_edges()
    chosen_indices = np.random.choice(
        flat_pairs_idx,
        size=num_excess_links,
        replace=False,
        p=probabilities
    )

    # translate flat indices back to (i, j) node coordinates and add to graph
    chosen_sources = i_grid.ravel()[chosen_indices]
    chosen_targets = j_grid.ravel()[chosen_indices]

    new_edges = list(zip(chosen_sources, chosen_targets))
    G.add_edges_from(new_edges)

    # assign influence multiplier to all asymmetric edges
    for u, v in G.edges():
        if not (v, u) in G.edges():
            G[u][v]['influence_multiplier'] = alpha
        else:
            G[u][v]['influence_multiplier'] = 1

    return G
