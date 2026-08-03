import numpy as np
import networkx as nx
from numpy.typing import NDArray
from scipy.sparse.linalg import eigs, spsolve, cg
from scipy.sparse import csr_matrix, diags, eye as speye
from scipy.sparse.linalg import ArpackNoConvergence


def build_weight_matrix(G: nx.DiGraph, alpha: float) -> csr_matrix:
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

    Parameters
    ----------
    G : nx.DiGraph
        Graph produced by generator.create_new(), with an 'is_asymmetric'
        attribute on every edge.
    alpha : float
        Influence multiplier applied to a junior node's weighting of the
        senior node its edge points to.

    Returns
    -------
    W : (N, N) sparse CSR row-stochastic matrix
        W[i,j] is the weight agent i places on agent j's opinion.
    """
    N = G.number_of_nodes()
    rows, cols, vals = [], [], []
    for u, v in G.edges():
        if G[u][v]['is_asymmetric']:
            rows += [u, v]
            cols += [v, u]
            vals += [1.0, alpha]  # u (senior) weighs v at 1; v (junior) weighs u at alpha
        else:
            rows.append(u)
            cols.append(v)
            vals.append(1.0)

    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    A = A + speye(N, format='csr')  # self-weight = 1

    row_sums = np.asarray(A.sum(axis=1)).ravel()
    assert np.all(
        row_sums > 0), "Every node must have at least itself as neighbor"
    W = diags(1.0 / row_sums) @ A

    return W.tocsr()


def trophic_coherence(G: nx.DiGraph) -> dict:
    """
    Compute trophic levels and the incoherence parameter q for a GPPM
    graph (Johnson et al. 2014, "Trophic coherence determines food-web
    stability").

    Trophic level s_i is defined by minimizing sum over edges (i,j) of
    (s_i - s_j - 1)^2 -- i.e. every edge "should" span a level gap of 1,
    matching this graph's convention (source is one level more senior
    than target) -- subject to s_i = 1 for basal nodes. Setting the
    gradient to zero gives a sparse linear system in the free (non-basal)
    variables, solved directly. The incoherence parameter q is the
    standard deviation of the actual per-edge level gaps s_i - s_j; q = 0
    means every edge spans exactly one level (perfectly coherent).

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

    k_out = np.zeros(N)
    k_in = np.zeros(N)
    rows, cols, vals = [], [], []
    for u, v in G.edges():
        k_out[u] += 1
        k_in[v] += 1
        rows += [u, v]
        cols += [v, u]
        vals += [1.0, 1.0]
    A_sym = csr_matrix((vals, (rows, cols)), shape=(N, N))
    deg_total = k_out + k_in

    s = np.ones(N)
    if len(free) > 0:
        A_free_free = A_sym[free, :][:, free]
        A_free_basal = A_sym[free, :][:, basal]
        L_matrix = (diags(deg_total[free]) - A_free_free).tocsr()
        rhs = (k_out - k_in)[free] + np.asarray(
            A_free_basal.sum(axis=1)).ravel()

        # L_matrix is a symmetric, grounded graph Laplacian (positive
        # definite as long as every free node reaches a basal node) --
        # CG converges in a fraction of a second where SuperLU's default
        # ordering in spsolve can take a minute on graphs this size.
        s_free, info = cg(L_matrix, rhs, rtol=1e-10)
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
        Initial opinions, typically drawn from Uniform(0, 1).
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


def run_trial(G: nx.DiGraph, alpha: float) -> dict:
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

    Returns
    -------
    dict combining outputs of compute_spectral_gap() and
    trophic_coherence().
    """
    W = build_weight_matrix(G, alpha)
    spectral_results = compute_spectral_gap(W)
    trophic_results = trophic_coherence(G)

    return {
        **spectral_results,
        **trophic_results,
    }
