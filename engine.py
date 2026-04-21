import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix, diags, eye as speye
from scipy.sparse.linalg import ArpackNoConvergence


def build_weight_matrix(
    adjacency: NDArray[np.float64],
    node_weights: NDArray[np.float64],
) -> csr_matrix:
    """
    Construct a row-stochastic DeGroot weight matrix from an adjacency
    matrix and node influence weights. Returns a sparse CSR matrix.

    Parameters
    ----------
    adjacency : (N, N) binary array
        Unweighted adjacency matrix. A[i,j] = 1 if i and j are connected.
        Should have zero diagonal (no self-loops in the topology).
        Can be directed or undirected. Dense or sparse.
    node_weights : (N,) array
        Influence weight for each node. Must be strictly positive.

    Returns
    -------
    W : (N, N) sparse CSR row-stochastic matrix
        W[i,j] is the weight agent i places on agent j's opinion.
        Each row sums to 1. W[i,i] > 0 for all i (self-inclusion).
    """
    N = adjacency.shape[0]
    assert adjacency.shape == (N, N), "Adjacency matrix must be square"
    assert node_weights.shape == (N,), "Node weights must match adjacency size"
    assert np.all(
        node_weights > 0), "All node weights must be strictly positive"

    # convert to sparse if dense, then add self-loops
    A = csr_matrix(adjacency, dtype=np.float64)
    A = A + speye(N, format='csr')

    # apply node influence weights: multiply each column j by w[j]
    # right-multiplication by diagonal = column scaling
    W = A @ diags(node_weights)

    # row-normalize to make stochastic
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    assert np.all(
        row_sums > 0), "Every node must have at least itself as neighbor"
    W = diags(1.0 / row_sums) @ W

    return W.tocsr()


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


def run_trial(
    adjacency: NDArray[np.float64],
    node_weights: NDArray[np.float64],
    rng: np.random.Generator | None = None,
    max_steps: int = 10_000,
    threshold: float = 1e-6,
) -> dict:
    """
    Convenience function: build weight matrix, run one simulation trial
    with random initial opinions, and compute spectral gap.

    Parameters
    ----------
    adjacency : (N, N) binary array
    node_weights : (N,) array
    rng : numpy random Generator (for reproducibility)
    max_steps : int
    threshold : float

    Returns
    -------
    dict combining outputs of simulate_degroot() and compute_spectral_gap(),
    plus initial opinions x0.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = adjacency.shape[0]
    W = build_weight_matrix(adjacency, node_weights)
    x0 = rng.uniform(0.0, 1.0, size=N)

    sim_results = simulate_degroot(
        W, x0, max_steps=max_steps, threshold=threshold)
    spectral_results = compute_spectral_gap(W)

    return {
        'x0': x0,
        **sim_results,
        **spectral_results,
    }
