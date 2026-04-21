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


def make_weights(
    N: int,
    weight_dist: Optional[str],
    rng: np.random.Generator,
    sigma: float = 1.2,
    S: Optional[float] = None,
    depth: Optional[NDArray] = None,
    max_depth: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Generate node influence weights.  Shared by all topologies.

    Parameters
    ----------
    N : int
        Number of nodes.
    weight_dist : str or None
        None        -> all ones (equal influence)
        'uniform'   -> U(1, 10)
        'normal'    -> 1 + |N(0, sigma)|
        'skewed'  -> mean 0
        'stepped'   -> deterministic linear-in-depth (hierarchy only;
                       flat topologies fall back to all ones)
    rng : Generator
    a : float
        Pareto shape parameter (default 2.0).
    sigma : float
        Half-normal scale (default 5.0).
    S : float or None
        Step size for 'stepped' mode. If None, auto-computed so that
        root weight / leaf weight ≈ 2*sigma + 1 (matching the ~95th
        percentile of the half-normal distribution used by 'normal').
    depth : array, optional
        Per-node depth in the tree (0 = root). Provided only by the
        hierarchy branch.
    max_depth : int, optional
        Maximum depth of the tree.

    Returns
    -------
    weights : (N,) float64 array, all positive.

    Notes
    -----
    For the stochastic distributions ('uniform', 'normal', 'powerlaw')
    on hierarchy (when depth is provided): weights are drawn iid from
    the target distribution, then sorted so that higher-level nodes
    receive higher weights.  This preserves the marginal distribution
    (same histogram as the flat topologies) while correlating influence
    with structural position.
    """
    # --- trivial case: equal weights ---
    if weight_dist is None:
        return np.full(N, 1.0)

    # --- deterministic depth-based weights ---
    if weight_dist == 'stepped':
        if depth is None or max_depth is None or max_depth == 0:
            return np.full(N, 1.0)
        reversed_depth = max_depth - depth
        if S is None:
            # auto-scale: root ≈ 1 + 2*sigma, leaf = 1
            S = (2.0 * sigma) / max_depth
        return 1.0 + reversed_depth.astype(np.float64) * S

    # --- stochastic distributions ---
    if weight_dist == 'uniform':
        w = rng.uniform(1.0, 10.0, N)
    elif weight_dist == 'normal':
        w = 1.0 + np.abs(rng.standard_normal(N)) * sigma
    elif weight_dist == 'skewed':
        w = rng.lognormal(mean=0.0, sigma=sigma, size=N)
    else:
        raise ValueError(
            f"Unknown weight_dist '{weight_dist}'. "
            "Expected None, 'uniform', 'normal', 'powerlaw', or 'stepped'."
        )

    # for hierarchy: sort so higher-level nodes get higher weights
    if depth is not None and max_depth is not None:
        sorted_w = np.sort(w)[::-1]  # descending values
        reversed_depth = max_depth - depth
        rank_order = np.argsort(-reversed_depth, kind='stable')  # root first
        out = np.empty(N, dtype=np.float64)
        out[rank_order] = sorted_w
        return out

    return w


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

    N is assumed to already be a valid tree size.

    Args
    ---
    graph_type : str
        "control"       - random regular graph (exact degree C)
        "alternative"   - Watts-Strogatz small-world graph
        "hierarchy"     - b-ary tree with probabilistic leaf-to-leaf edges
    C : int or float
        Target mean degree.  Must be even (Watts-Strogatz requirement).
    N : int
        Size of the graph. Should already be a valid tree size.
    weight_dist : str or None
        Node weight distribution (shared by all topologies):
        None        -> all ones (equal influence)
        'uniform'   -> U(1, 10)
        'normal'    -> 1 + |N(0, sigma)|
        'powerlaw'  -> Pareto(a) + 1
        'stepped'   -> deterministic by tree depth (hierarchy only)
    rng : numpy.random.Generator, optional
    **kwargs :
        a (float):        Pareto shape (default 2.0)
        sigma (float):    half-normal scale (default 5.0)
        S (float):        stepped weight step size; auto-computed if None
        p_rewire (float): WS rewiring probability (default 0.1)
        Pd (float):       hierarchy edge probability decay (default 0.5)

    Returns
    ---
    adj : (N, N) binary ndarray, undirected, no self-loops
    weights : (N,) ndarray of node influence weights (all positive)
    '''
    if rng is None:
        rng = np.random.default_rng()

    N = int(N)

    # unpack kwargs
    a = kwargs.get('a', 2.0)
    sigma = kwargs.get('sigma', 1.2)
    S = kwargs.get('S', None)
    p_rewire = kwargs.get('p_rewire', 0.1)
    Pd = kwargs.get('Pd', 0.5)

    b = int(C - 1)  # branching factor

    # shared weight kwargs for flat topologies (no depth info)
    wt_kw = dict(weight_dist=weight_dist, rng=rng, sigma=sigma)

    # -----------------------------------------------------------------
    # CONTROL: random regular graph (every node has exactly degree C)
    # -----------------------------------------------------------------
    if graph_type == 'control':
        G = nx.random_regular_graph(int(C), N, seed=rng)
        adj = nx.to_numpy_array(G)
        return adj, make_weights(N, **wt_kw)  # ty: ignore

    # -----------------------------------------------------------------
    # ALTERNATIVE: Watts-Strogatz small-world
    # -----------------------------------------------------------------
    elif graph_type == 'alternative':
        k = int(C)
        G = nx.watts_strogatz_graph(N, k, p_rewire, seed=rng)
        adj = nx.to_numpy_array(G)
        adj = _ensure_connected(adj, rng)
        return adj, make_weights(N, **wt_kw)  # ty: ignore

    # -----------------------------------------------------------------
    # HIERARCHY: b-ary tree + relatedness-weighted leaf edges
    # -----------------------------------------------------------------
    elif graph_type == 'hierarchy':

        # --- build tree structure arithmetically ---
        parent_arr = np.full(N, -1, dtype=np.intp)
        depth = np.zeros(N, dtype=np.intp)
        for n in range(1, N):
            parent_arr[n] = (n - 1) // b
            depth[n] = depth[parent_arr[n]] + 1

        max_depth = int(depth.max())

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

        # --- add leaf-to-leaf edges to reach target mean degree ---
        tree_edges = N - 1
        target_edges = int(C * N / 2)
        n_needed = max(0, target_edges - tree_edges)

        new_rows = []
        new_cols = []

        if n_needed > 0:
            leaves_under = {}
            for lf in leaves:
                leaves_under[int(lf)] = [int(lf)]
            for n in range(N - 1, -1, -1):
                if has_child[n] and n not in leaves_under:
                    acc = []
                    for c in range(n * b + 1, min(n * b + b + 1, N)):
                        acc.extend(leaves_under.get(c, []))
                    leaves_under[n] = acc

            internal_nodes = []
            sampling_weights = []
            child_leaf_lists = []

            for v in np.where(has_child)[0]:
                children = list(range(v * b + 1, min(v * b + b + 1, N)))
                cl = [leaves_under.get(c, []) for c in children]
                cl = [lst for lst in cl if len(lst) > 0]
                if len(cl) < 2:
                    continue

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

            edge_set = set()
            for n in range(1, N):
                p_n = parent_arr[n]
                edge_set.add((min(n, p_n), max(n, p_n)))

            batch = max(n_needed * 3, 1000)
            max_attempts = 20

            for _ in range(max_attempts):
                if len(new_rows) >= n_needed:
                    break
                sampled = rng.choice(
                    len(internal_nodes), size=batch, replace=True,
                    p=sampling_weights
                )
                for idx in sampled:
                    if len(new_rows) >= n_needed:
                        break
                    cl = child_leaf_lists[idx]
                    sizes = np.array([len(lst) for lst in cl], dtype=float)
                    p1 = sizes / sizes.sum()
                    ci = rng.choice(len(cl), p=p1)
                    s2 = sizes.copy()
                    s2[ci] = 0
                    if s2.sum() == 0:
                        continue
                    s2 /= s2.sum()
                    cj = rng.choice(len(cl), p=s2)
                    u = int(rng.choice(cl[ci]))
                    v = int(rng.choice(cl[cj]))
                    edge = (min(u, v), max(u, v))
                    if edge not in edge_set:
                        edge_set.add(edge)
                        new_rows.append(edge[0])
                        new_cols.append(edge[1])

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

        # --- weights: use shared make_weights with depth info ---
        node_weights = make_weights(
            N, weight_dist=weight_dist, rng=rng,
            sigma=sigma, S=S,
            depth=depth, max_depth=max_depth,
        )

        return adj, node_weights

    else:
        raise ValueError(
            f"Unsupported graph type: {graph_type}. "
            "Expected 'control', 'alternative', or 'hierarchy'."
        )
