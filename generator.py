import numpy as np
import networkx as nx
from itertools import combinations
from numpy.typing import NDArray
from typing import Tuple, Optional


def _ensure_connected(adj: NDArray, rng: np.random.Generator) -> NDArray:
    """Add minimal random edges to connect all components."""
    G = nx.from_numpy_array(adj)
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return adj
    adj = adj.copy()
    for i in range(len(components) - 1):
        u = rng.choice(sorted(components[i]))
        v = rng.choice(sorted(components[i + 1]))
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def generator(
    graph_type: str,
    C: int | float,
    N: int | np.integer,
    weight_dist: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    **kwargs
) -> Tuple[NDArray, NDArray]:
    '''
    Generate a graph of type `graph_type` with target mean degree C and size N.

    Args
    ---
    graph_type : str
        Must be either:
        - "control" (Erdos-Renyi random graph)
        - "hierarchy" (Tree with branching factor C-1, descending weights of
          step size S, and further edges probabilistically added between leaves
          based on relatedness to bring mean degree in line with C)
        - "alternative" (Degree-corrected SBM with n_comms communities and
          in/out connection probabilities i_prob and o_prob)
    C : int or float
        Target mean degree.
    N : int
        Size of the graph.
    weight_dist : str, optional
        Distribution for node influence weights (non-hierarchy types):
        - "uniform" (Uniform(0, 10))
        - "powerlaw" (Pareto with shape "a")
        - "normal" (half-normal centered at 1 with std dev "sigma")
        - None (all weights = 1)
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.
    **kwargs : Keyword arguments:
        - a (float): Pareto shape parameter (default 2.0)
        - sigma (float): half-normal scale (default 5.0)
        - i_prob (float): SBM within-community edge factor (default 0.8)
        - o_prob (float): SBM between-community edge factor (default 0.2)
        - n_comms (int): number of SBM communities (default 8)
        - S (int): hierarchy weight step size (default 1)

    Returns
    ---
    adj : (N, N) binary array
        Adjacency matrix (undirected, no self-loops).
    weights : (N,) array
        Node influence weights.
    '''
    if rng is None:
        rng = np.random.default_rng()

    N = int(N)

    # unpack kwargs
    a = kwargs.get('a', 2.0)
    scale = kwargs.get('sigma', 5.0)
    i_prob = kwargs.get('i_prob', 0.8)
    o_prob = kwargs.get('o_prob', 0.2)
    n_comms = kwargs.get('n_comms', 8)
    S = kwargs.get('S', 1)

    b = C - 1  # branching factor

    # N must be restricted to multiples of branching factor starting from the nearest full size in order for 
    # hierarchical structure to maintain C. This function finds the closest valid size to chosen N.
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
        print(f"N rounded up to {N_new} to maintain C.") if N < N_new else print(f"N rounded down to {N_new} to maintain C.")

    N = N_new

    # dynamically set std dev of deg dist
    deg_sigma = max(C / 30, 1.0)

    # --- non-hierarchy types ---
    if graph_type != 'hierarchy':
        # generate node weights
        if weight_dist == 'uniform':
            weights = rng.uniform(0.0, 10.0, N)
        elif weight_dist == 'powerlaw':
            weights = rng.pareto(a, N)
        elif weight_dist == 'normal':
            weights = 1.0 + np.abs(rng.standard_normal(N)) * scale
        else:
            weights = np.full(N, 1.0)

        if graph_type == 'control':
            p = C / N
            G = nx.erdos_renyi_graph(N, p, seed=rng)
            adj = nx.to_numpy_array(G)
            adj = _ensure_connected(adj, rng)
            return adj, weights

        elif graph_type == 'alternative':
            def build_alt():
                exp_deg = rng.normal(C, deg_sigma, N)
                exp_deg = np.clip(exp_deg, 1, None)

                # community assignments
                comm_size = max(1, N // n_comms)
                comm = np.arange(N) // comm_size
                same_comm = comm[:, None] == comm[None, :]
                factor = np.where(same_comm, i_prob, o_prob)

                # edge probabilities
                probs = np.outer(exp_deg, exp_deg) * factor / (C * N)

                # undirected: draw upper triangle only, then mirror
                upper = np.triu(rng.random((N, N)) < probs, k=1).astype(np.float64)
                return upper + upper.T

            adj = build_alt()
            adj = _ensure_connected(adj, rng)
            return adj, weights

        else:
            raise ValueError(
                f"Unsupported graph type: {graph_type}. "
                "Expected 'control', 'hierarchy', or 'alternative'."
            )

    # --- hierarchy ---
    else:
        def build_tree(S):
            G = nx.DiGraph()
            G.add_node(0)
            queue = [0]
            next_id = 1
            while next_id < N:
                parent = queue.pop(0)
                for _ in range(b): # type: ignore
                    if next_id >= N:
                        break
                    G.add_edge(parent, next_id)
                    queue.append(next_id)
                    next_id += 1

            leaves = {n for n, degree in G.out_degree() if degree == 0}

            # --- node:level mapping ---
            levels_dict = nx.shortest_path_length(G, source=0)
            max_dist = max(levels_dict.values())
            nodes_to_level = {node: (max_dist - dist) for node, dist in levels_dict.items()}

            # --- add edges between leaves to reach mean degree C ---
            Pd = 0.5  # probability decay parameter
            candidates = [(u, v) for u, v in combinations(leaves, 2)
                          if not G.has_edge(u, v)]

            # weight by hierarchical distance (closer relatives more likely)
            edge_probs = np.array([
                Pd ** nodes_to_level[nx.lowest_common_ancestor(G, u, v)]
                for u, v in candidates
            ])
            edge_probs /= edge_probs.sum()

            # compute how many edges to add to reach target mean degree
            tree_edges = N - 1
            target_edges = int(C * N / 2)
            n_needed = max(0, target_edges - tree_edges)
            n_needed = min(n_needed, len(candidates))

            chosen_indices = rng.choice(
                len(candidates), size=n_needed, replace=False, p=edge_probs
            )

            for idx in chosen_indices:
                u, v = candidates[idx]
                G.add_edge(u, v)

            # convert to undirected adjacency matrix
            G_u = G.to_undirected()
            adj = nx.to_numpy_array(G_u, nodelist=sorted(G.nodes()))

            # --- descending weights by level ---
            level_to_weight = {}
            current_weight = 1
            for level in sorted(set(nodes_to_level.values())):
                level_to_weight[level] = current_weight
                current_weight += S

            node_weights = np.array([
                level_to_weight[nodes_to_level[node]]
                for node in sorted(nodes_to_level.keys())
            ])

            return adj, node_weights

        return build_tree(S)