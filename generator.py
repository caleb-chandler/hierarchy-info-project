import numpy as np
import networkx as nx
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from typing import Tuple, Optional


def valid_tree_sizes(b: int, N_max: int) -> NDArray[np.int64]:
    """
    Compute all valid tree sizes up to N_max for branching factor b.
    Exported so simulation.py can build N_range once from these.
    """
    candidates = set()
    h = 1
    while True:
        base = int((b**(h+1) - 1) / (b - 1))
        for k in range(0, b**h + 1):
            val = base + k * b
            if val <= N_max:
                candidates.add(val)
        if base > 2 * N_max:
            break
        h += 1
    return np.sort(np.array(list(candidates), dtype=np.int64))


def snap_to_valid(b: int, N: int) -> int:
    """Round N to the nearest valid tree size for branching factor b."""
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

    N is assumed to already be a valid tree size (use snap_to_valid or
    valid_tree_sizes to pre-compute valid sizes).

    Args
    ---
    graph_type : str
        "control", "hierarchy", or "alternative"
    C : int or float
        Target mean degree.
    N : int
        Size of the graph. Should already be a valid tree size.
    weight_dist : str, optional
        Node weight distribution for non-hierarchy types:
        "uniform", "powerlaw", "normal", or None (all ones).
    rng : numpy.random.Generator, optional
    **kwargs :
        a (float): Pareto shape (default 2.0)
        sigma (float): half-normal scale (default 5.0)
        i_prob (float): SBM within-community factor (default 0.8)
        o_prob (float): SBM between-community factor (default 0.2)
        n_comms (int): SBM community count (default 8)
        S (int): hierarchy weight step size (default 1)
        Pd (float): hierarchy edge probability decay (default 0.5)

    Returns
    ---
    adj : (N, N) binary ndarray, undirected, no self-loops
    weights : (N,) ndarray of node influence weights
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
    Pd = kwargs.get('Pd', 0.5)

    b = int(C - 1)  # branching factor
    deg_sigma = max(C / 30, 1.0)

    # --- weight generation (shared by control & alternative) ---
    def make_weights(N):
        if weight_dist == 'uniform':
            return rng.uniform(0.0, 10.0, N)
        elif weight_dist == 'powerlaw':
            return rng.pareto(a, N)
        elif weight_dist == 'normal':
            return 1.0 + np.abs(rng.standard_normal(N)) * scale
        else:
            return np.full(N, 1.0)

    # -----------------------------------------------------------------
    # CONTROL: Erdos-Renyi
    # -----------------------------------------------------------------
    if graph_type == 'control':
        p = C / N
        G = nx.erdos_renyi_graph(N, p, seed=rng)
        adj = nx.to_numpy_array(G)
        adj = _ensure_connected(adj, rng)
        return adj, make_weights(N)

    # -----------------------------------------------------------------
    # ALTERNATIVE: degree-corrected SBM
    # -----------------------------------------------------------------
    elif graph_type == 'alternative':
        exp_deg = rng.normal(C, deg_sigma, N)
        exp_deg = np.clip(exp_deg, 1, None)

        comm_size = max(1, N // n_comms)
        comm = np.arange(N) // comm_size
        same_comm = comm[:, None] == comm[None, :]
        factor = np.where(same_comm, i_prob, o_prob)

        raw_probs = np.outer(exp_deg, exp_deg) * factor
        # scale so expected total edges = C*N/2 (target mean degree C)
        upper_raw = np.triu(raw_probs, k=1)
        scale = (C * N / 2) / upper_raw.sum()
        probs = np.clip(raw_probs * scale, 0, 1)

        # undirected: draw upper triangle only, then mirror
        upper = np.triu(rng.random((N, N)) < probs, k=1).astype(np.float64)
        adj = upper + upper.T
        adj = _ensure_connected(adj, rng)
        return adj, make_weights(N)

    # -----------------------------------------------------------------
    # HIERARCHY: b-ary tree + relatedness-weighted leaf edges
    # -----------------------------------------------------------------
    elif graph_type == 'hierarchy':

        # --- build tree structure arithmetically ---
        # BFS-ordered b-ary tree: parent(n) = (n-1) // b
        parent_arr = np.full(N, -1, dtype=np.intp)
        depth = np.zeros(N, dtype=np.intp)
        for n in range(1, N):
            parent_arr[n] = (n - 1) // b
            depth[n] = depth[parent_arr[n]] + 1

        max_depth = int(depth.max())

        # identify leaves (nodes whose first potential child >= N)
        has_child = np.zeros(N, dtype=bool)
        has_child[parent_arr[1:]] = True
        leaves = np.where(~has_child)[0]

        # --- tree adjacency (undirected, sparse) ---
        child_ids = np.arange(1, N)
        parent_ids = parent_arr[1:]
        rows = np.concatenate([parent_ids, child_ids])
        cols = np.concatenate([child_ids, parent_ids])
        tree_adj = csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(N, N)
        )

        # --- add edges between leaves to reach target mean degree ---
        tree_edges = N - 1
        target_edges = int(C * N / 2)
        n_needed = max(0, target_edges - tree_edges)

        new_rows = []
        new_cols = []

        if n_needed > 0:
            # precompute leaf descendants per node (bottom-up)
            leaves_under = {}
            for lf in leaves:
                leaves_under[int(lf)] = [int(lf)]
            for n in range(N - 1, -1, -1):
                if has_child[n] and n not in leaves_under:
                    acc = []
                    for c in range(n * b + 1, min(n * b + b + 1, N)):
                        acc.extend(leaves_under.get(c, []))
                    leaves_under[n] = acc

            # two-level sampling setup:
            # for each internal node v, compute
            #   weight = Pd^(max_depth - depth[v]) * num_cross_subtree_leaf_pairs
            internal_nodes = []
            sampling_weights = []
            child_leaf_lists = []  # cl[i] = list of leaf-arrays per child subtree

            for v in np.where(has_child)[0]:
                children = list(range(v * b + 1, min(v * b + b + 1, N)))
                cl = [leaves_under.get(c, []) for c in children]
                cl = [lst for lst in cl if len(lst) > 0]
                if len(cl) < 2:
                    continue

                # count cross-subtree pairs
                sizes = [len(lst) for lst in cl]
                total_pairs = 0
                for i in range(len(sizes)):
                    for j in range(i + 1, len(sizes)):
                        total_pairs += sizes[i] * sizes[j]

                if total_pairs == 0:
                    continue

                w = Pd ** (max_depth - depth[v]) * total_pairs
                internal_nodes.append(int(v))
                sampling_weights.append(w)
                child_leaf_lists.append(cl)

            sampling_weights = np.array(sampling_weights)
            sampling_weights /= sampling_weights.sum()

            # collect existing edges to avoid duplicates
            edge_set = set()
            for n in range(1, N):
                p_n = parent_arr[n]
                edge_set.add((min(n, p_n), max(n, p_n)))

            # sample edges: pick LCA node, then two leaves from different subtrees
            batch = max(n_needed * 3, 1000)
            max_attempts = 20

            for _ in range(max_attempts):
                if len(new_rows) >= n_needed:
                    break
                sampled = rng.choice(
                    len(internal_nodes), size=batch, replace=True, p=sampling_weights
                )
                for idx in sampled:
                    if len(new_rows) >= n_needed:
                        break
                    cl = child_leaf_lists[idx]
                    sizes = np.array([len(lst) for lst in cl], dtype=float)
                    # pick first subtree weighted by size
                    p1 = sizes / sizes.sum()
                    ci = rng.choice(len(cl), p=p1)
                    # pick second subtree from remainder
                    s2 = sizes.copy()
                    s2[ci] = 0
                    if s2.sum() == 0:
                        continue
                    s2 /= s2.sum()
                    cj = rng.choice(len(cl), p=s2)
                    # pick one leaf from each subtree
                    u = int(rng.choice(cl[ci]))
                    v = int(rng.choice(cl[cj]))
                    edge = (min(u, v), max(u, v))
                    if edge not in edge_set:
                        edge_set.add(edge)
                        new_rows.append(edge[0])
                        new_cols.append(edge[1])

        # assemble full adjacency
        if new_rows:
            extra_r = np.array(new_rows + new_cols)
            extra_c = np.array(new_cols + new_rows)
            extra = csr_matrix(
                (np.ones(len(extra_r), dtype=np.float64), (extra_r, extra_c)),
                shape=(N, N)
            )
            full_adj = tree_adj + extra
        else:
            full_adj = tree_adj

        adj = np.asarray(full_adj.todense()).clip(0, 1)

        # --- descending weights by level ---
        # reversed depth: leaves=0, root=max_depth
        reversed_depth = max_depth - depth
        level_to_weight = {}
        current_weight = 1
        for level in range(max_depth + 1):
            level_to_weight[level] = current_weight
            current_weight += S

        node_weights = np.array(
            [level_to_weight[reversed_depth[n]] for n in range(N)],
            dtype=np.float64
        )

        return adj, node_weights

    else:
        raise ValueError(
            f"Unsupported graph type: {graph_type}. "
            "Expected 'control', 'hierarchy', or 'alternative'."
        )