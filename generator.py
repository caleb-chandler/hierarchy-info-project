import numpy as np
import networkx as nx
from engine import build_weight_matrix
from scipy.stats import halfnorm
import math
from itertools import combinations
import random

def generator(type, C, N, weight_dist=None, **kwargs):
    ''' 
    Function to generate a graph of type "type" with mean C of standard normal
    degree-distribution and size N.

    Args
    ---
    type : str
        Must be either:
            - "control" (G,n,p random)
            - "hierarchy" (Tree with branching factor C, descending weights of step size S, and
            further edges probabilistically added between leaves based on relatedness to bring
            mean degree in line with C)
            - "alternative" (Custom degree-controlled SBM with # communities
            n_comms and in/out connection probabilities i_prob and o_prob)
    C : int or float
        Center of degree distribution.
    N : int
        Size of the graph.
    weight_dist : str
        Type of distribution to draw weights from if heterogeneity desired
        and type is not hierarchy. Options:
            - "uniform" (standard uniform distribution between 0 and 10)
            - "powerlaw" (power-law distribution with decay "a")
            - "normal" (right-tailed normal distribution centered at 1 with std dev "scale")
    **kwargs : Keyword arguments passed to:
        - np.random.pareto (a = 2)
        - halfnorm.rvs (scale = 5)
        - build_alt (i_prob = 0.2, o_prob = 0.8, n_comms = 8)
        - build_tree (S = 1)

    Returns
    ---
    graph : (N, N) row-stochastic array
        Normalized weighted adjacency matrix.
    '''
    # unpacking kwargs
    a = kwargs.get('a', 2.0)
    scale = kwargs.get('sigma', 5.0)
    i_prob = kwargs.get('i_prob', 0.8)
    o_prob = kwargs.get('o_prob', 0.2)
    n_comms = kwargs.get('n_comms', 8)
    S = kwargs.get('S', 1)

    b = C-1 # branching factor

    # N must be restricted to multiples of branching factor starting from the nearest full size in order for 
    # hierarchical structure to maintain C. This function finds the closest valid size to chosen N
    def size_transformer(b, N):
        h = 1
        candidates = []
        while True:
            base = int((b**(h+1) - 1) / (b - 1))
            if base - b**h * b > N + N:  # gone far enough
                break
            for k in range(0, b**h + 1):
                candidates.append(base + k * b)
            h += 1
        return min(candidates, key=lambda x: abs(x - N))

    # replace N with nearest valid size
    N_new = size_transformer(b, N)
    if N != N_new:
        print(f"N rounded up to {N_new} to maintain C...") if N < N_new else print(f"N rounded down to {N_new} to maintain C...")

    N = N_new

    # dynamically setting std dev of deg dist
    deg_sigma = C/30 if C/30 >= 1 else 1

    if type != 'hierarchy':
        # assign weights and build array
        if weight_dist == 'uniform':
            weights = np.random.uniform(0.0, 10.0, N)
        elif weight_dist == 'powerlaw':
            weights = np.random.pareto(a, N)
        elif weight_dist == 'normal':
            weights = halfnorm.rvs(loc=1.0, scale=scale, size=N)
        else:
            weights = np.full(N, 1.0)

        if type == 'control':
            p = C / N # enforcing degree distribution
            G = nx.erdos_renyi_graph(N, p)
            adj = nx.to_numpy_array(G)
            return build_weight_matrix(adj, weights)

        elif type == 'alternative':
            def build_alt(i_prob=i_prob, o_prob=o_prob, n_comms=n_comms):
                '''
                Implements degree-corrected SBM to enforce weight distribution
                '''
                # calculate expected degrees
                exp_deg = np.random.normal(C, deg_sigma, N)
                exp_deg = np.clip(exp_deg, 1, None) # ensure deg >= 1

                probs = np.zeros((N, N))
                for i in range(N):
                    for j in range(i + 1, N):
                        # floor division to assign nodes to equally-sized communities
                        comm_i = i // int((N/n_comms))
                        comm_j = j // int((N/n_comms))
                        # assign probs accordingly
                        factor = i_prob if comm_i == comm_j else o_prob
                        # calculate probability of edge between node i and j and normalize
                        probs[i, j] = (exp_deg[i] * exp_deg[j] * factor) / (C * N)
                # mirror to fill in both triangles
                probs = probs + probs.T
                # flip coin for every possible edge simultaneously and assign 0 or 1
                return (np.random.rand(N, N) < probs).astype(np.float64)
            adj = build_alt()
            return build_weight_matrix(adj, weights)

        else:
            print("Error: Invalid type.")
            return None
    
    else:
        def build_tree(S):
            b = C-1 # branching factor

            # building initial graph
            G = nx.DiGraph() # init as directed for easier lookup
            G.add_node(0) # root
            queue = [0]
            next_id = 1
            while next_id < N:
                parent = queue.pop(0)
                for _ in range(b): # repeat for all children
                    if next_id >= N:
                        break
                    G.add_edge(parent, next_id)
                    queue.append(next_id)
                    next_id += 1

            # leaves have deg 0 in directed case
            leaves = {n for n, degree in G.out_degree() if degree == 0}
            branches = {n for n, degree in G.out_degree() if degree > 0}

             # --- branch:leaf descendants mapping ---

            # recursive function to find all leaf descendants
            memo = {}
            def get_leaf_descendants(node):
                if node in memo:
                    return memo[node]
                
                # if node is a leaf, it is its own descendant
                if node in leaves:
                    return {node}
                
                # recursive case: goes down the chain until it finds the leaves
                all_leaves = set()
                for child in G.successors(node):
                    all_leaves.update(get_leaf_descendants(child))
                
                memo[node] = all_leaves
                return all_leaves

            # generate the final mapping
            branch_to_leaves = {b: list(get_leaf_descendants(b)) for b in branches}

            # --- node:level mapping ---

            levels_dict = nx.shortest_path_length(G, source=0)
            # reversing order so that probability calculation works correctly
            max_dist = max(levels_dict.values())
            nodes_to_level = {node: (max_dist - dist) for node, dist in levels_dict.items()}

            h = int(max(levels_dict.values())) # store height as int

            # --- adding new edges between leaves ---

            # determining probabilities
            Pd = 0.5 # probability decay parameter (controls shape of distribution)
            def edge_prob(i, j):
                LCA = nx.lowest_common_ancestor(G, i, j)
                LCA_level = nodes_to_level[LCA]
                return Pd**LCA_level
            
            # adding only enough edges to match density
            sample = random.sample(list(combinations(leaves, 2)), b**h/2)
            for u, v in sample:
                if G.has_edge(u, v) == False: # skip existing edges
                    if random.random() < edge_prob(u, v):
                        G.add_edge(u, v)

            # extracting adj matrix
            adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes())) # ensuring order

            # --- applying weights ---

            # weight-to-level dict
            level_to_weight = {}
            current_weight = 1
            for level in sorted(set(nodes_to_level.values())): # reversed the order previously, so it should be leaves -> root
                level_to_weight[level] = current_weight
                current_weight += S

            # assign node weights accordingly
            node_to_weight = {}
            for node, level in nodes_to_level.items():
                node_to_weight[node] = level_to_weight[level]
            
            # convert to array (sorting by node for consistency with adj)
            weights = np.array([w for n, w in sorted(node_to_weight.items())])

            return adj, weights
        
        adj, w = build_tree(S)
        return build_weight_matrix(adj, w)