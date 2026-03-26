import numpy as np
import networkx as nx
from scipy.stats import halfnorm
from itertools import combinations
import random
from numpy.typing import NDArray
from typing import Tuple, Optional

def generator(
    graph_type: str, 
    C: int | float, 
    N: int | np.integer,
    rng: np.random.Generator,
    weight_dist: Optional[str] = None,
    **kwargs
) -> Tuple[NDArray, NDArray]:
    ''' 
    Function to generate a graph of type "graph_type" with mean C of standard normal
    degree-distribution and size N.

    Args
    ---
    graph_type : str
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
    (N, N) binary array
        Adjacency matrix.
    (N,) vector array
        Node weights.
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
            for k in range(0, b**h + 1):
                candidates.append(base + k * b)
            if base > 2 * N:
                break
            h += 1
        return min(candidates, key=lambda x: abs(x - N))

    # replace N with nearest valid size
    N_new = size_transformer(b, N)
    if N != N_new:
        print(f"N rounded up to {N_new} to maintain C...") if N < N_new else print(f"N rounded down to {N_new} to maintain C...")

    N = N_new

    # dynamically setting std dev of deg dist
    deg_sigma = C/30 if C/30 >= 1 else 1

    if graph_type != 'hierarchy':
        # assign weights and build array
        if weight_dist == 'uniform':
            weights = np.random.uniform(0.0, 10.0, N)
        elif weight_dist == 'powerlaw':
            weights = np.random.pareto(a, N)
        elif weight_dist == 'normal':
            weights = halfnorm.rvs(loc=1.0, scale=scale, size=N)
        else:
            weights = np.full(N, 1.0)

        if graph_type == 'control':
            G = nx.erdos_renyi_graph(N, C / (N-1), seed=rng_integers)
            # connect components if disconnected
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                u = rng.choice(list(components[i]))
                v = rng.choice(list(components[i + 1]))
                G.add_edge(u, v)
            adj = nx.to_numpy_array(G)
            return adj, weights

        elif graph_type == 'alternative':
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
                upper = np.triu(np.random.rand(N, N) < probs, k=1)
                adj = (upper + upper.T).astype(np.float64)
                return adj
            adj = build_alt()
            return adj, weights

        else:
            raise ValueError(f"Unsupported graph type: {type}. Expected 'control', 'hierarchy', or 'alternative'.")
    
    else:
        def build_tree(S):
            b = C-1 # branching factor

            # building initial graph
            G = nx.DiGraph()
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

            # leaves have out-deg 0
            leaves = {n for n, degree in G.out_degree() if degree == 0}

            # --- node:level mapping ---

            levels_dict = nx.shortest_path_length(G, source=0)
            # reversing order so that probability calculation works correctly
            max_dist = max(levels_dict.values())
            nodes_to_level = {node: (max_dist - dist) for node, dist in levels_dict.items()}

            h = int(max(levels_dict.values())) # store height as int

            # --- adding new edges between leaves ---

            # determining probabilities
            Pd = 0.5 # probability decay parameter (controls shape of distribution)
            # all non-sibling leaf pairs as candidates
            candidates = [(u, v) for u, v in combinations(leaves, 2)
                        if not G.has_edge(u, v)]

            # weight each by hierarchical distance
            weights = np.array([
                Pd ** nodes_to_level[nx.lowest_common_ancestor(G, u, v)]
                for u, v in candidates
            ])

            # normalize to probability distribution
            weights /= weights.sum()

            # draw exactly the right number without replacement
            n_needed = int(b**h // 2)
            chosen_indices = np.random.choice(
                len(candidates), size=n_needed, replace=False, p=weights
            )

            # add the chosen edges
            for idx in chosen_indices:
                u, v = candidates[idx]
                G.add_edge(u, v)

            # converting back to undirected and extracting adj matrix
            G_u = G.to_undirected()
            adj = nx.to_numpy_array(G_u, nodelist=sorted(G.nodes())) # ensuring order

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
            weights = np.array([w for _, w in sorted(node_to_weight.items())])

            return adj, weights
        
        return build_tree(S)