"""
Model for the 2026-11-04 internal-consensus scaling run: rule B on a geometric pyramid whose depth
follows the top-size rule, with disorder added either through the level-gap kernel's width T or
through a rewiring fraction p. Self-contained: ablation/ablation_model.py, ruleB_overnight/model.py
and engine.build_weight_matrix are imported unchanged.

Levels: geometric, each level R = 0.65 times the size of the one below. Depth is the most levels
for which the top level still holds at least TOP = 15 nodes (5/8/10/13/16/19 levels at
N = 300/1,000/3,000/10,000/30,000/100,000), so the top tier stays a fixed-size group as N grows.

Edges (rule B): every non-basal node draws exactly C = 6 distinct subordinates.
  'T' family  each draw picks a level with probability ~ kernel x available nodes, with kernel
              exp(-(l_i - l_j - 1)^2 / (2 T^2)), then a node in it (ablation_model.rule_b_on_levels).
              T = inf makes the kernel flat, so subordinates are uniform over all nodes.
  'p' family  each draw is, with probability 1 - p, a node from the level directly below (the
              T = 0.1 kernel, which falls back to the nearest level with nodes left), and with
              probability p a node drawn uniformly from all nodes. This is the pure hierarchy with
              the lower end of each tie rewired to a random node with probability p.
  p = 0 is the same model as T = 0.1, and p = 1 the same as T = inf.

Influence: the alpha rule (engine.build_weight_matrix): a junior weights its senior (the tie's
authority end) at alpha, the senior weights the junior at 1, every self-tie is 1.

Outcome: continuous-time internal consensus, dx/dt = (W - I) x. Every mode decays at rate
1 - Re(lambda), so the time to delta = 1e-6 is ln(1e6) / (1 - Re lambda_+), where lambda_+ is W's
eigenvalue with the largest real part after 1.
"""
import os
import sys

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)
if os.path.join(PROBE, 'ablation') not in sys.path:
    sys.path.insert(0, os.path.join(PROBE, 'ablation'))
import ablation_model as am                                # noqa: E402  (unchanged)
from ablation_model import mdl, build_weight_matrix        # noqa: E402  (unchanged)

R, TOP, C, T_MIN = 0.65, 15, 6, 0.1
LN_1E6 = np.log(1e6)


def depth(N):
    '''The most levels for which the top level still holds at least TOP nodes.'''
    L = 1
    while True:
        try:
            if mdl.level_sizes(N, L + 1, R)[-1] < TOP:
                return L
        except ValueError:                                 # a level would be empty
            return L
        L += 1


def level_counts(N):
    return mdl.level_sizes(N, depth(N), R)


def rule_b_rewired(k, c, p, rng):
    '''Rule B where each subordinate is, with probability p, a uniformly random node instead of
    one from the level directly below. Node ids are grouped by level, as in rule_b_on_levels.'''
    k = np.asarray(k)
    n_levels, N = len(k), int(k.sum())
    lv = np.repeat(np.arange(n_levels), k)
    start = np.r_[0, np.cumsum(k)]
    L = np.arange(n_levels)
    log_kernel = -((L[:, None] - L[None, :] - 1) ** 2) / (2 * T_MIN ** 2)

    src, dst = [], []
    for i in range(start[1], N):
        s = lv[i]
        avail = k.astype(float)
        avail[s] -= 1
        chosen = set()
        for _ in range(c):
            if rng.random() < p:                           # rewired: any node
                while True:
                    j = int(rng.integers(N))
                    if j != i and j not in chosen:
                        break
                t = lv[j]
            else:                                          # hierarchical: rule B at T = 0.1
                with np.errstate(divide='ignore'):
                    log_mass = log_kernel[s] + np.log(avail)
                cdf = np.cumsum(np.exp(log_mass - log_mass.max()))
                t = np.searchsorted(cdf, rng.random() * cdf[-1], side='right')
                while True:
                    j = start[t] + rng.integers(k[t])
                    if j != i and j not in chosen:
                        break
            chosen.add(j)
            avail[t] -= 1
            src.append(i)
            dst.append(j)

    G = nx.DiGraph()
    G.add_nodes_from((n, {'level': int(lv[n]) + 1}) for n in range(N))
    G.add_edges_from(zip(src, dst))
    return G


def make_graph(family, param, N, rng):
    k = level_counts(N)
    if family == 'T':
        return am.rule_b_on_levels(k, C, param, rng)
    return rule_b_rewired(k, C, param, rng)


def tie_stats(G):
    '''Share of ties going exactly one level down, share going sideways or up (the senior end at
    or below the junior end's level), and mean |level gap - 1|, from the generator's levels.'''
    lv = nx.get_node_attributes(G, 'level')
    gap = np.array([lv[u] - lv[v] for u, v in G.edges()])
    return dict(frac_one_down=(gap == 1).mean(), frac_not_down=(gap <= 0).mean(),
                mean_gap_dev=np.abs(gap - 1).mean())


def influence(Gc, alpha, node_sets, tol=1e-10, check_every=25, max_iter=100_000):
    '''Stationary influence pi (pi W = pi, summing to 1) of the alpha-rule W, by lazy power iteration
    pi <- (pi + pi W) / 2. Every term is nonnegative, so tiny entries keep full relative precision;
    an eigensolver's error is relative to the largest entry, which loses the bottom levels' influence
    once it falls below ~1e-12. Stops when the summed influence of every set in node_sets changes by
    less than tol (relative) over check_every iterations. Returns pi, iterations, converged.'''
    WT = build_weight_matrix(Gc, alpha).T.tocsr()
    n = WT.shape[0]
    pi = np.full(n, 1 / n)
    last = np.array([pi[S].sum() for S in node_sets])
    for it in range(1, max_iter + 1):
        pi = 0.5 * (pi + WT @ pi)
        if it % check_every == 0:
            pi /= pi.sum()
            now = np.array([pi[S].sum() for S in node_sets])
            if np.all(np.abs(now - last) <= tol * now):
                return pi, it, True
            last = now
    return pi / pi.sum(), max_iter, False


def alignment_gap(Gc, alpha, sources, gamma):
    '''1 - rho for the real-node block Q of the alpha-rule W with the environment node. Q is
    nonnegative, so rho is its Perron root, and continuous-time alignment time is
    ln(1e6) / (1 - rho). Only trustworthy while 1 - rho is well above the solver's ~1e-13 error.'''
    W = build_weight_matrix(Gc, alpha, sources=sources, gamma=gamma)
    n = Gc.number_of_nodes()
    Q = W[:n, :n]
    vals = eigs(Q, k=1, which='LM', ncv=min(n - 1, 30), tol=1e-13, maxiter=100_000,
                v0=np.ones(n), return_eigenvectors=False)
    return 1 - abs(vals[0])


def continuous_time(Gc, alpha):
    '''Continuous-time internal consensus time and lambda_+ for the alpha-rule W.'''
    W = build_weight_matrix(Gc, alpha)
    n = W.shape[0]
    vals = eigs(W, k=3, which='LR', ncv=min(n - 1, 60), tol=1e-10, maxiter=100_000,
                v0=np.ones(n) + np.linspace(0, 1, n), return_eigenvectors=False)
    vals = vals[np.argsort(-vals.real)]
    assert abs(vals[0] - 1) < 1e-8, vals[0]
    return LN_1E6 / (1 - vals[1].real), vals[1]
