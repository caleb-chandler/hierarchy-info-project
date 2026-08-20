import numpy as np
import networkx as nx


def calibrate_density(b, N_max, margin=9.0):
    '''
    Recommend a constant density ratio c (= L/N) that keeps a GPPM graph
    connected (beyond a small residual of orphaned basal nodes) across an
    entire ensemble of sizes up to N_max, using a single fixed c for every
    N in the sweep.

    Excess ratio $R_e =\frac{M(B_m\cdot\log_{B_m})}{N_m}$ or in other words 
    the number of trials (excess links) expected before each possible sample 
    (isolated tree) is drawn per $N$ at the largest $N$ (with added buffer $M$)

    1-b is equivalent to the non-basal node ratio via "tree_edges = N-B = N-bN"
    and "tree_edges_per_node = N-bN/N = 1-b." This is added to the excess ratio
    to obtain the total ratio which is c. Only works if 1 edge is added at a time

    Args:
        b: basal fraction (B = round(b*N)).
        N_max: largest N that will appear in the ensemble.
        margin: safety multiplier over the raw B*ln(B) threshold.

    Returns:
        Recommended value of c (= L/N) to pass to create_new for every N.
    '''
    B_max = max(1, round(b * N_max))
    if B_max <= 1:
        excess_ratio = 0.0
    else:
        excess_ratio = margin * B_max * np.log(B_max) / N_max
    return (1.0 - b) + excess_ratio


def create_new(N, b, c, T, rng):
    '''
    Generate a random hierarchical influence network using the Generalized
    Preferential Preying Model (GPPM).

    Builds a tree of N-B non-basal nodes attached uniformly at random on top
    of B basal (level-1) nodes, then adds excess links until the graph has L
    edges total. Excess link endpoints are sampled without replacement, with
    probability weighted by a Gaussian kernel over level difference (peaked
    at a level gap of 1, with bandwidth T), favoring links between nodes one
    level apart. Every edge is tagged 'is_asymmetric': True if only one
    direction between its endpoints was drawn (no reciprocal edge exists),
    False if both directions were drawn. Actual influence weighting (e.g.
    an asymmetry multiplier alpha) is applied downstream, since it doesn't
    depend on which edges exist -- only on which ones are asymmetric.

    Args:
        N: total number of nodes.
        b: basal fraction; B = round(b*N) basal (level-1) nodes are
            created, and the remaining N-B are attached above them.
        c: target density; L = round(c*N) is the total edge count (must be
            >= N-B, the number of tree edges added during initialization).
            Use calibrate_density() to pick a c that stays connected across
            an entire N-ensemble.
        T: bandwidth of the Gaussian level-difference kernel used to weight
            excess-link sampling; smaller values concentrate links more
            tightly around a level gap of 1.
        rng: numpy.random.Generator used for all random draws.

    Returns:
        A networkx.DiGraph with N nodes, L edges, a 'level' attribute on
        each node (basal nodes have level == 1), and an 'is_asymmetric'
        attribute on each edge.
    '''
    B = max(1, round(b * N))
    L = round(c * N)
    if L < N - B:
        raise ValueError(
            f"c={c} gives L={L}, below the {N - B} tree edges required "
            f"for N={N}, b={b}"
        )

    # --- init base layer and attrs ---
    G = nx.DiGraph()
    G.add_nodes_from(range(B), level=1)
    non_basal = list(range(B, N))

    # --- begin addition process ---

    # create list once and append to it for faster runtime
    available_nodes = list(G.nodes())
    tree_edges = []
    for n in non_basal:
        subordinate = available_nodes[rng.integers(len(available_nodes))]
        G.add_edge(n, subordinate)
        tree_edges.append((n, subordinate))
        G.nodes[n]['level'] = G.nodes[subordinate]['level'] + 1
        available_nodes.append(n)

    del available_nodes  # no longer needed
    tree_edge_set = set(tree_edges)

    # --- add excess links ---

    # Weight depends only on (source_level, target_level), and tree depth
    # grows just ~logarithmically with N, so there are only O(log N) levels
    # -- far fewer than N. Bucketing by level-pair keeps the accounting
    # O(D^2) (D = number of distinct levels) instead of O(N*(N-B)), which
    # is what blows up memory/time for large N.
    levels = np.array([G.nodes[n]['level'] for n in range(N)])
    non_basal_ids = np.arange(B, N)

    # Basal nodes have no prey (edges point authority -> subordinate, so
    # "prey of i" means i's out-neighbors) -- excluded as excess-edge
    # sources, not targets, matching the tree-building step above, where
    # only non-basal nodes ever originate an edge.
    sources_by_level = {
        lvl: non_basal_ids[levels[non_basal_ids] == lvl]
        for lvl in np.unique(levels[non_basal_ids])
    }
    targets_by_level = {
        lvl: np.where(levels == lvl)[0] for lvl in np.unique(levels)
    }

    bucket_sl, bucket_tl, bucket_weight, bucket_count = [], [], [], []
    for sl, src_ids in sources_by_level.items():
        for tl, tgt_ids in targets_by_level.items():
            count = len(src_ids) * len(tgt_ids)
            if sl == tl:
                count -= len(tgt_ids)  # exclude self-loops
            if tl == sl - 1:
                count -= len(src_ids)  # exclude existing tree edges
            if count <= 0:
                continue
            bucket_sl.append(sl)
            bucket_tl.append(tl)
            bucket_weight.append(np.exp(-((sl - tl - 1) ** 2) / (2 * T ** 2)))
            bucket_count.append(count)

    bucket_sl = np.array(bucket_sl)
    bucket_tl = np.array(bucket_tl)
    bucket_mass = np.array(bucket_weight) * np.array(bucket_count)
    total_mass = bucket_mass.sum()
    if total_mass == 0:
        print("Error: Weights not added.")
        return G
    bucket_probs = bucket_mass / total_mass

    # sample excess links without replacement: draw (bucket, then a
    # uniform pair within that bucket, since all pairs in a bucket share
    # the same weight) with replacement, rejecting collisions/exclusions,
    # topping up as needed. Collisions are rare since num_excess_links is
    # tiny relative to the total candidate population.
    num_excess_links = L - G.number_of_edges()
    chosen_pairs = set()
    max_attempts = 200
    for _ in range(max_attempts):
        if len(chosen_pairs) >= num_excess_links:
            break
        n_needed = num_excess_links - len(chosen_pairs)
        n_draw = int(n_needed * 1.2) + 10
        draw_buckets = rng.choice(
            len(bucket_probs), size=n_draw, p=bucket_probs)
        for bi in draw_buckets:
            sl, tl = bucket_sl[bi], bucket_tl[bi]
            src_ids, tgt_ids = sources_by_level[sl], targets_by_level[tl]
            s = src_ids[rng.integers(len(src_ids))]
            t = tgt_ids[rng.integers(len(tgt_ids))]
            if s == t or (s, t) in tree_edge_set or (s, t) in chosen_pairs:
                continue
            chosen_pairs.add((s, t))
            if len(chosen_pairs) >= num_excess_links:
                break
    else:
        raise RuntimeError(
            "Could not sample enough distinct excess links; "
            "L may be too close to the total number of valid pairs."
        )

    G.add_edges_from(chosen_pairs)

    # tag every edge as asymmetric (no reciprocal edge) or not
    for u, v in G.edges():
        G[u][v]['is_asymmetric'] = not G.has_edge(v, u)

    return G


def largest_connected_component(G):
    '''
    Restrict G to its largest connected component (checked on the
    undirected projection), dropping any smaller fragments -- including
    the basal nodes that can end up fully isolated when the tree-building
    step never selects them as a target. Nodes are relabeled to a
    contiguous 0..n-1 range; basal-ness should be checked afterward via
    the 'level' attribute (level == 1), not by node id.

    Args:
        G: a graph produced by create_new().

    Returns:
        (G_clean, n_dropped): the induced subgraph on the largest
        connected component, and the number of nodes dropped.
    '''
    components = nx.connected_components(G.to_undirected())
    giant = max(components, key=len)
    n_dropped = G.number_of_nodes() - len(giant)
    G_clean = nx.convert_node_labels_to_integers(
        G.subgraph(giant).copy(), ordering='sorted'
    )
    return G_clean, n_dropped
