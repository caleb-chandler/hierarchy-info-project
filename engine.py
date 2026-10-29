import numpy as np
import networkx as nx
from numpy.typing import NDArray
from scipy.sparse.linalg import eigs, spsolve, bicgstab
from scipy.sparse import csr_matrix, diags, eye as speye
from scipy.sparse.linalg import ArpackNoConvergence


def build_weight_matrix(
    G: nx.DiGraph,
    alpha: float,
    sources: NDArray[np.int_] | None = None,
    gamma: float | None = None,
) -> csr_matrix:
    """
    Construct a row-stochastic DeGroot weight matrix from a GPPM graph.

    Influence is bidirectional between any two connected nodes -- edge
    direction only marks which side is amplified. The edge (u, v) itself
    is the operational definition of "u is senior to v" for that tie
    (regardless of the two nodes' 'level' values, which govern generation
    but not this weighting): for an edge tagged 'is_asymmetric' (no
    reciprocal edge exists), the junior v weights the senior u at
    `alpha`, and u weights v at baseline 1. Edges that already have an
    explicit reciprocal get weight 1 in both directions. Every node also
    weights itself at 1 (self-loop).

    If `sources` is given, an environment node is appended at index N.
    It models external information the group should align with: its row
    is only its self-loop, so its opinion never changes, and each source
    node i gives it exactly a `gamma` share of its row: with R_i the raw
    weight i places on everything else (self-loop plus neighbors), the
    environment gets raw weight R_i * gamma/(1-gamma), so after row
    normalization W[i, env] = gamma and i's other weights keep their
    proportions, scaled by (1-gamma). The environment node is the only
    absorbing state, so
    |λ₂| of W is the spectral radius of the real-node block, and
    compute_spectral_gap's predicted time becomes the time for every node
    to come within δ of the environment's opinion. G itself (and hence
    trophic levels and q) is untouched.

    Parameters
    ----------
    G : nx.DiGraph
        Graph produced by generator.create_new(), with an 'is_asymmetric'
        attribute on every edge.
    alpha : float
        Influence multiplier applied to a junior node's weighting of the
        senior node its edge points to.
    sources : array of int, optional
        Node ids (0..N-1) that receive input from the environment node.
        None (default) omits the environment node entirely.
    gamma : float in (0, 1)
        Share of each source node's attention given to the environment
        node. Required when `sources` is given.

    Returns
    -------
    W : (N, N) sparse CSR row-stochastic matrix, or (N+1, N+1) with the
        environment node as the last row/column when `sources` is given.
        W[i,j] is the weight agent i places on agent j's opinion.
    """
    N = G.number_of_nodes()
    rows, cols, vals = [], [], []
    for u, v in G.edges():
        if G[u][v]['is_asymmetric']:
            rows += [u, v]
            cols += [v, u]
            # u (senior) weighs v at 1; v (junior) weighs u at alpha
            vals += [1.0, alpha]
        else:
            rows.append(u)
            cols.append(v)
            vals.append(1.0)

    size = N
    if sources is not None:
        sources = np.unique(np.asarray(sources, dtype=int))
        if len(sources) == 0:
            raise ValueError(
                "sources is empty; pass None to omit the environment node")
        if sources[0] < 0 or sources[-1] >= N:
            raise ValueError(f"source ids must lie in 0..{N - 1}")
        if gamma is None or not 0 < gamma < 1:
            raise ValueError(
                f"gamma must lie in (0, 1) when sources are given, got {gamma}")
        # raw row sum before the env entry: neighbors plus the self-loop
        # of 1 added below
        rest = 1 + np.bincount(np.asarray(rows, dtype=int),
                               weights=vals, minlength=N)
        # sizing the env entry at rest * gamma/(1-gamma) makes it exactly
        # a gamma share of the normalized row; the environment's own row
        # gets only the self-loop added below
        rows += sources.tolist()
        cols += [N] * len(sources)
        vals += (rest[sources] * gamma / (1 - gamma)).tolist()
        size = N + 1

    A = csr_matrix((vals, (rows, cols)), shape=(size, size))
    A = A + speye(size, format='csr')  # self-weight = 1

    row_sums = np.asarray(A.sum(axis=1)).ravel()
    assert np.all(
        row_sums > 0), "Every node must have at least itself as neighbor"
    W = diags(1.0 / row_sums) @ A

    return W.tocsr()


def trophic_coherence(G: nx.DiGraph) -> dict:
    """
    Compute trophic levels and the incoherence parameter q for a GPPM
    graph, using the standard food-web prey-averaged formula: basal
    nodes (no prey) are fixed at s_i = 1, and every other node's level
    is s_i = 1 + mean(s_j for j among i's prey).

    Edges in this graph point from authority to subordinate (edge (i,j)
    means i has authority over j), which is the reverse of the typical
    ecological convention (edges prey -> predator, so a predator's prey
    are its in-neighbors). Here, "prey of i" is i's out-neighbors --
    whichever nodes i has authority over -- so the averaging is over
    out-neighbors, normalized by out-degree, not in-degree.

    This gives a directed linear system in the free (non-basal)
    variables (basal levels are known constants), solved directly. The
    incoherence parameter q is the standard deviation of the actual
    per-edge level gaps s_i - s_j; q = 0 means every edge spans exactly
    one level (perfectly coherent), and the mean gap is always 1 by
    construction (a node's own level is forced to average 1 above its
    prey's).

    Parameters
    ----------
    G : nx.DiGraph
        Graph produced by generator.create_new() (or
        generator.largest_connected_component() applied to one), with a
        'level' node attribute (basal nodes have level == 1).

    Returns
    -------
    dict with keys:
        'trophic_levels' : (N,) array, s_i for each node.
        'mean_trophic_distance' : float, mean of s_i - s_j over edges.
        'trophic_incoherence' : float, q = std of s_i - s_j over edges.
    """
    N = G.number_of_nodes()
    is_basal = np.array(
        [G.nodes[n]['level'] == 1 for n in range(N)], dtype=bool)
    basal = np.where(is_basal)[0]
    free = np.where(~is_basal)[0]

    out_degree = np.zeros(N)
    rows, cols = [], []
    for u, v in G.edges():
        out_degree[u] += 1
        rows.append(u)
        cols.append(v)
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))

    s = np.ones(N)
    if len(free) > 0:
        A_free_free = A[free, :][:, free]
        A_free_basal = A[free, :][:, basal]
        L_matrix = (diags(out_degree[free]) - A_free_free).tocsr()
        rhs = out_degree[free] + np.asarray(
            A_free_basal.sum(axis=1)).ravel()

        # L_matrix is diagonally dominant (out_degree counts every
        # out-edge, A_free_free only the subset landing on free nodes)
        # but not symmetric -- unlike the old undirected-Laplacian
        # version, so CG doesn't apply; BiCGSTAB handles the general
        # case and still converges fast on this well-conditioned system.
        s_free, info = bicgstab(L_matrix, rhs, rtol=1e-10)
        if info != 0:
            s_free = spsolve(L_matrix, rhs)
        s[free] = s_free

    x = np.array([s[u] - s[v] for u, v in G.edges()])

    return {
        'trophic_levels': s,
        'mean_trophic_distance': float(x.mean()),
        'trophic_incoherence': float(x.std()),
    }


def simulate_degroot(
    W: csr_matrix,
    x0: NDArray[np.float64],
    max_steps: int = 10_000,
    threshold: float = 1e-6,
) -> dict:
    """
    Run DeGroot dynamics until consensus or max steps reached.

    Parameters
    ----------
    W : (N, N) sparse row-stochastic matrix
        Weight matrix from build_weight_matrix().
    x0 : (N,) array
        Initial opinions, typically drawn from Uniform(0, 1). If W has an
        environment node, its fixed opinion is the last entry, and
        max(x) - min(x) then bounds every node's distance from it.
    max_steps : int
        Safety cap on iterations.
    threshold : float
        Convergence criterion: stop when max(x) - min(x) < threshold.

    Returns
    -------
    dict with keys:
        'converged' : bool
            Whether consensus was reached within max_steps.
        'consensus_time' : int
            Number of steps to reach consensus (max_steps if not converged).
        'final_opinions' : (N,) array
            Opinion vector at termination.
        'final_disagreement' : float
            max(x) - min(x) at termination.
        'consensus_value' : float
            Mean of final opinions (= the consensus if converged).
        'disagreement_history' : list[float]
            max(x) - min(x) at each step, for diagnostics/plotting.
    """
    N = W.shape[0]
    assert x0.shape == (N,), "Initial opinions must match weight matrix size"

    x = x0.copy()
    disagreement_history = []

    for t in range(max_steps):
        disagreement = x.max() - x.min()
        disagreement_history.append(disagreement)

        if disagreement < threshold:
            return {
                'converged': True,
                'consensus_time': t,
                'final_opinions': x,
                'final_disagreement': disagreement,
                'consensus_value': x.mean(),
                'disagreement_history': disagreement_history,
            }

        x = W @ x  # sparse @ dense vector -> dense vector

    # Final check after last update
    disagreement = x.max() - x.min()
    disagreement_history.append(disagreement)

    return {
        'converged': disagreement < threshold,
        'consensus_time': max_steps,
        'final_opinions': x,
        'final_disagreement': disagreement,
        'consensus_value': x.mean(),
        'disagreement_history': disagreement_history,
    }


def compute_spectral_gap(W: csr_matrix) -> dict:
    """
    Compute the spectral gap of the weight matrix, which determines
    the asymptotic rate of convergence.

    For a row-stochastic matrix with a unique stationary distribution,
    the largest eigenvalue is 1. The spectral gap is 1 - |λ₂|, where
    λ₂ is the second-largest eigenvalue in modulus.

    Convergence time scales as ~ 1 / spectral_gap (up to log factors).
    More precisely, time to reach threshold δ ≈ log(1/δ) / log(1/|λ₂|).

    Uses ARPACK sparse eigensolver by default. Falls back to dense
    eigensolver if ARPACK fails to converge (common with poorly
    conditioned matrices from highly skewed weight distributions).

    Parameters
    ----------
    W : (N, N) sparse row-stochastic matrix

    Returns
    -------
    dict with keys:
        'eigenvalues' : (2,) complex array
            Two largest eigenvalues by modulus.
        'lambda_2_modulus' : float
            |λ₂|, the second-largest eigenvalue modulus.
        'spectral_gap' : float
            1 - |λ₂|.
        'predicted_convergence_time' : float
            Estimated steps to reach threshold 1e-6, computed as
            log(1e-6) / log(|λ₂|). Returns inf if |λ₂| >= 1.
        'used_dense_fallback' : bool
            True if ARPACK failed and dense eigensolver was used.
    """
    if not isinstance(W, csr_matrix):
        W = csr_matrix(W)

    used_dense = False

    try:
        eigenvalues, _ = eigs(W, k=2, which='LM', maxiter=10000)
    except (ArpackNoConvergence, RuntimeError):
        # fall back to dense eigensolver
        used_dense = True
        W_dense = W.toarray()
        all_eigs = np.linalg.eigvals(W_dense)
        # pick the two largest by modulus
        moduli_all = np.abs(all_eigs)
        top2_idx = np.argsort(-moduli_all)[:2]
        eigenvalues = all_eigs[top2_idx]

    # sort by modulus, descending
    moduli = np.abs(eigenvalues)
    sorted_indices = np.argsort(-moduli)
    eigenvalues_sorted = eigenvalues[sorted_indices]
    moduli_sorted = moduli[sorted_indices]

    lambda_2_mod = float(moduli_sorted[1])
    spectral_gap = 1.0 - lambda_2_mod

    if lambda_2_mod < 1.0:
        predicted_time = np.log(1e-6) / np.log(lambda_2_mod)
    else:
        predicted_time = np.inf

    return {
        'eigenvalues': eigenvalues_sorted,
        'lambda_2_modulus': lambda_2_mod,
        'spectral_gap': spectral_gap,
        'predicted_convergence_time': predicted_time,
        'used_dense_fallback': used_dense,
    }


def run_trial(
    G: nx.DiGraph,
    alpha: float,
    sources: NDArray[np.int_] | None = None,
    gamma: float | None = None,
) -> dict:
    """
    Convenience function: build the weight matrix for a GPPM graph at a
    given alpha, and compute its spectral gap and trophic coherence.

    Parameters
    ----------
    G : nx.DiGraph
        Graph produced by generator.create_new() (should already be
        restricted to its largest connected component).
    alpha : float
        Influence multiplier for asymmetric edges.
    sources : array of int, optional
        Nodes seeded by the environment node (see build_weight_matrix).
        With sources, the predicted convergence time is the time to align
        with the environment rather than to internal consensus.
    gamma : float in (0, 1)
        Share of each source node's attention given to the environment
        node (see build_weight_matrix). Required when `sources` is given.

    Returns
    -------
    dict combining outputs of compute_spectral_gap() and
    trophic_coherence().
    """
    W = build_weight_matrix(G, alpha, sources=sources, gamma=gamma)
    spectral_results = compute_spectral_gap(W)
    trophic_results = trophic_coherence(G)

    return {
        **spectral_results,
        **trophic_results,
    }
