"""
DeGroot Consensus Engine
========================
Core dynamics engine for studying how network topology and heterogeneous
influence affect consensus formation in social groups.

The update rule is standard DeGroot:
    x(t+1) = W @ x(t)

where W is a row-stochastic weight matrix encoding both topology (who
communicates with whom) and influence (how much weight agent i gives
to agent j's opinion).

Weight matrix construction:
    Given an adjacency matrix A (binary, no self-loops) and a vector of
    node influence weights w, we construct W as follows:
        - Agent i's neighborhood includes itself plus all j where A[i,j] = 1
        - The raw weight of neighbor j on agent i is w[j]
        - The raw self-weight of agent i is w[i]
        - Row-normalize so each row sums to 1

    So W[i,j] = w[j] / sum(w[k] for k in N(i) ∪ {i})  if j in N(i) ∪ {i}
       W[i,j] = 0                                       otherwise

This means high-influence nodes pull their neighbors' opinions more strongly.
Self-inclusion ensures aperiodicity (necessary for convergence on bipartite
graphs like trees).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix, diags, eye as speye


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
    assert np.all(node_weights > 0), "All node weights must be strictly positive"

    # convert to sparse if dense, then add self-loops
    A = csr_matrix(adjacency, dtype=np.float64)
    A = A + speye(N, format='csr')

    # apply node influence weights: multiply each column j by w[j]
    # right-multiplication by diagonal = column scaling
    W = A @ diags(node_weights)

    # row-normalize to make stochastic
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    assert np.all(row_sums > 0), "Every node must have at least itself as neighbor"
    W = diags(1.0 / row_sums) @ W

    return W.tocsr() # type: ignore


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
    N = W.shape[0] # type: ignore
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
    """
    if not isinstance(W, csr_matrix):
        W = csr_matrix(W)

    eigenvalues, _ = eigs(W, k=2)  # type: ignore[misc]

    # Sort by modulus, descending
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

    sim_results = simulate_degroot(W, x0, max_steps=max_steps, threshold=threshold)
    spectral_results = compute_spectral_gap(W)

    return {
        'x0': x0,
        **sim_results,
        **spectral_results,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("DeGroot Engine Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Test 1: Complete graph, N=10, uniform weights
    print("\n--- Test 1: Complete graph, N=10, uniform weights ---")
    N = 10
    A = np.ones((N, N)) - np.eye(N)
    w = np.ones(N)
    result = run_trial(A, w, rng=rng)
    print(f"  Spectral gap:      {result['spectral_gap']:.6f}")
    print(f"  |λ₂|:             {result['lambda_2_modulus']:.6f}")
    print(f"  Predicted time:    {result['predicted_convergence_time']:.1f}")
    print(f"  Actual time:       {result['consensus_time']}")
    print(f"  Converged:         {result['converged']}")
    print(f"  Final disagreement: {result['final_disagreement']:.2e}")

    # Test 2: Path graph, N=10, uniform weights
    print("\n--- Test 2: Path graph, N=10, uniform weights ---")
    A = np.zeros((N, N))
    for i in range(N - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    w = np.ones(N)
    result = run_trial(A, w, rng=rng)
    print(f"  Spectral gap:      {result['spectral_gap']:.6f}")
    print(f"  |λ₂|:             {result['lambda_2_modulus']:.6f}")
    print(f"  Predicted time:    {result['predicted_convergence_time']:.1f}")
    print(f"  Actual time:       {result['consensus_time']}")
    print(f"  Converged:         {result['converged']}")
    print(f"  Final disagreement: {result['final_disagreement']:.2e}")

    # Test 3: Star graph, N=10, center has high weight
    print("\n--- Test 3: Star graph, N=10, center weight=5, leaves=1 ---")
    A = np.zeros((N, N))
    for i in range(1, N):
        A[0, i] = 1
        A[i, 0] = 1
    w = np.ones(N)
    w[0] = 5.0
    result = run_trial(A, w, rng=rng)
    print(f"  Spectral gap:      {result['spectral_gap']:.6f}")
    print(f"  |λ₂|:             {result['lambda_2_modulus']:.6f}")
    print(f"  Predicted time:    {result['predicted_convergence_time']:.1f}")
    print(f"  Actual time:       {result['consensus_time']}")
    print(f"  Converged:         {result['converged']}")
    print(f"  Final disagreement: {result['final_disagreement']:.2e}")

    # Test 4: Verify simulation matches spectral prediction
    print("\n--- Test 4: Prediction accuracy across 5 random graphs ---")
    for trial in range(5):
        N = 20
        A = np.zeros((N, N))
        perm = rng.permutation(N)
        for i in range(N - 1):
            A[perm[i], perm[i + 1]] = 1
            A[perm[i + 1], perm[i]] = 1
        for _ in range(N):
            i, j = rng.integers(0, N, size=2)
            if i != j:
                A[i, j] = 1
                A[j, i] = 1
        w = rng.uniform(0.5, 2.0, size=N)
        result = run_trial(A, w, rng=rng)
        ratio = result['consensus_time'] / result['predicted_convergence_time']
        print(f"  Trial {trial + 1}: predicted={result['predicted_convergence_time']:.1f}, "
              f"actual={result['consensus_time']}, ratio={ratio:.3f}")

    print("\n" + "=" * 60)
    print("Self-test complete.")
    print("=" * 60)