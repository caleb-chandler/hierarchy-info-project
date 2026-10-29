import pickle
import numpy as np
from datetime import date
from pathlib import Path
from scipy.sparse.linalg import eigs, ArpackNoConvergence

import generator as gen
from engine import build_weight_matrix, compute_spectral_gap, trophic_coherence
from simulation import P, N_range, c

# Environmental-input tests. An environment node with a fixed opinion is
# attached to a set of source nodes, each giving it a GAMMA share of its
# attention; the outcome is the time for every node to align with it.
#   1. hierarchy: q = 0 graphs, sources = top / bottom SOURCE_FRAC of nodes
#      by trophic level s (plus a random SOURCE_FRAC on the same graphs)
#   2. random: sources = uniformly random SOURCE_FRAC of nodes, over
#      Q_RANGE (T calibrated per N exactly as in simulation.py)

# --- environment parameters ---
# small enough that alignment is set by where the sources sit (their total
# influence) rather than capped by the group's internal consensus speed:
# within ~5% of the influence-only rate for top/bottom/random at q = 0
GAMMA = 0.01         # share of each source's attention given to the environment
SOURCE_FRAC = 0.1    # fraction of nodes seeded; = b, so 'bottom' is the basal layer
T_HIERARCHY = 0.1    # every excess edge spans exactly one level, so q = 0
Q_RANGE = np.linspace(P.Q_RANGE[0], P.Q_RANGE[-1], 15)  # same span as simulation.py, coarser


def pick_sources(rule, s, rng):
    '''
    Choose SOURCE_FRAC of the nodes as environment sources.

    Args:
        rule: 'top' / 'bottom' (highest / lowest trophic level s, ties
            broken at random -- at q = 0 every s is an integer level, so
            the top picks are the highest levels with the boundary level
            sampled, and with SOURCE_FRAC = b the bottom picks are the
            basal layer, topped up by a few level-2 nodes if the LCC step
            dropped isolated basal nodes) or 'random' (uniform without
            replacement).
        s: (n,) trophic levels of the graph's nodes.
        rng: numpy.random.Generator for tie-breaking / sampling.

    Returns:
        Array of source node ids.
    '''
    n = len(s)
    k = max(1, round(SOURCE_FRAC * n))
    if rule == 'random':
        return rng.choice(n, size=k, replace=False)
    sign = -1 if rule == 'top' else 1
    return np.lexsort((rng.random(n), sign * s))[:k]


def influence(W):
    '''Stationary distribution (left Perron vector) of W: each node's long-run share of influence.'''
    try:
        _, vec = eigs(W.T, k=1, which='LM', maxiter=10000)
    except (ArpackNoConvergence, RuntimeError):
        return np.full(W.shape[0], np.nan)
    pi = np.abs(vec[:, 0])
    return pi / pi.sum()


def env_trial(G, rules, rng):
    '''
    Measure internal consensus and environmental alignment on one graph.

    Args:
        G: graph already restricted to its largest connected component.
        rules: source-selection rules to test, each on this same graph.
        rng: numpy.random.Generator for source selection.

    Returns:
        dict with the graph's q, internal consensus time (no environment
        node), and per rule the alignment time, |λ₂|, number of sources,
        and the sources' total influence in the network without the
        environment node.
    '''
    tc = trophic_coherence(G)
    W0 = build_weight_matrix(G, P.ALPHA)
    internal = compute_spectral_gap(W0)
    pi = influence(W0)

    align = {}
    for rule in rules:
        S = pick_sources(rule, tc['trophic_levels'], rng)
        res = compute_spectral_gap(
            build_weight_matrix(G, P.ALPHA, sources=S, gamma=GAMMA))
        align[rule] = {
            'conv_time': res['predicted_convergence_time'],
            'lambda_2_modulus': res['lambda_2_modulus'],
            'used_dense_fallback': res['used_dense_fallback'],
            'n_sources': len(S),
            'source_influence': float(pi[S].sum()),
        }

    return {
        'actual_q': tc['trophic_incoherence'],
        'internal_conv_time': internal['predicted_convergence_time'],
        'internal_lambda_2_modulus': internal['lambda_2_modulus'],
        'N_actual': G.number_of_nodes(),
        'align': align,
    }


def run_cell(N, T, rules, graph_rng, source_rng):
    trials = []
    for _ in range(P.M):
        G, _, tree_depth, _, _ = gen.create_new(
            N=int(N), b=P.B_FRACTION, c=c, T=T, rng=graph_rng)
        Gc, n_dropped = gen.largest_connected_component(G)
        trial = env_trial(Gc, rules, source_rng)
        trial.update({'T': T, 'n_dropped': n_dropped, 'tree_depth': tree_depth})
        trials.append(trial)

    med = {r: np.median([t['align'][r]['conv_time'] for t in trials]) for r in rules}
    print(f"  N={N:>6d} | median_q={np.median([t['actual_q'] for t in trials]):.4f}  "
          f"internal={np.median([t['internal_conv_time'] for t in trials]):.1f}  "
          + "  ".join(f"{r}={v:.1f}" for r, v in med.items()), flush=True)
    return trials


def main():
    graph_rng = np.random.default_rng(P.SEED + 2)
    source_rng = np.random.default_rng(P.SEED + 3)

    # --- output directory ---
    save_dir = Path('results') / date.today().isoformat() / 'env'
    k = 0
    while save_dir.exists():
        k += 1
        save_dir = save_dir.with_name(f'env_{k}')
    save_dir.mkdir(parents=True)

    print(f"N_range: {N_range.tolist()}")
    print(f"alpha={P.ALPHA}, gamma={GAMMA}, source fraction={SOURCE_FRAC}, "
          f"M={P.M}, c={c:.4f}", flush=True)

    # --- test 1: pure hierarchy, top vs bottom vs random sources ---
    print(f"\nHierarchy (T={T_HIERARCHY}, q=0): top / bottom / random sources", flush=True)
    rules = ['top', 'bottom', 'random']
    results_bag = {int(N): run_cell(N, T_HIERARCHY, rules, graph_rng, source_rng)
                   for N in N_range}
    with open(save_dir / 'hierarchy_q0.pkl', 'wb') as f:
        pickle.dump(results_bag, f)

    # --- test 2: random sources over the full q range ---
    # same calibration stream and call order as simulation.py, so the q(T)
    # tables are identical to the ones used there
    cal_rng = np.random.default_rng(P.SEED + 1)
    T_big = gen.calibrate_T(Q_RANGE, 1000, P.B_FRACTION, c, cal_rng)
    T_by_N = {N: gen.calibrate_T(Q_RANGE, N, P.B_FRACTION, c, cal_rng) if N < 1000 else T_big
              for N in N_range}
    T_lines = "\n".join(
        f"{N:>6d} : {np.round(T_by_N[N], 4).tolist()}" for N in N_range)
    print(f"\ncalibrated T (rows N, columns Q_RANGE):\n{T_lines}", flush=True)

    for i, q in enumerate(Q_RANGE):
        print(f"\nRandom sources, q={q:.4f}", flush=True)
        results_bag = {int(N): run_cell(N, T_by_N[N][i], ['random'], graph_rng, source_rng)
                       for N in N_range}
        q_label = f'{q:.4f}'.replace('.', 'p')
        with open(save_dir / f'random_q_{q_label}.pkl', 'wb') as f:
            pickle.dump(results_bag, f)

    content = f"""
```python
ALPHA : {P.ALPHA}
B_FRACTION : {P.B_FRACTION}
DENSITY_MARGIN : {P.DENSITY_MARGIN}
N_RANGE : {N_range.tolist()}
Q_RANGE : {Q_RANGE.tolist()}
M : {P.M}
GAMMA : {GAMMA}
SOURCE_FRAC : {SOURCE_FRAC}
T_HIERARCHY : {T_HIERARCHY}
SEEDS : graphs {P.SEED + 2}, sources {P.SEED + 3}, calibration {P.SEED + 1}
```

Calibrated T (rows N, columns Q_RANGE):
```
{T_lines}
```
"""
    with open(save_dir / 'params.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\nSaved to {save_dir}\nDone.")


if __name__ == '__main__':
    main()
