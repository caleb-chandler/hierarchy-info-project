"""
Model pieces for the 2026-10-29 overnight tests (rule-B pyramid + sort-assigned
influence weights + environment input). Self-contained: nothing in the existing
codebase is modified. Trophic levels and the largest-connected-component step
reuse engine.trophic_coherence and generator.largest_connected_component as-is.

Network (rule B):
    Levels 1..D hold a geometric pyramid of node counts (each level r times the
    one below, rounded to sum to N). Every non-basal node draws exactly c
    distinct subordinates (out-edges). Each candidate node j is weighted by a
    Gaussian kernel on the level gap, exp(-(l_i - l_j - 1)^2 / (2 T^2)), so a
    level's chance is kernel x the number of its nodes still available.

Influence (alpha removed):
    Ties are undirected with weight 1, every node also has a self-tie, and
    everything flowing into node j is scaled by its influence weight w_j:
        W = D^-1 K diag(w),   K = undirected adjacency + I,   d = K w.
    Weights are lognormal quantiles with median 1: the k-th smallest of n is
    exp(sigma * Phi^-1((k - 0.5) / n)). 'sorted' gives the largest weights to
    the highest trophic levels (ties broken at random); 'random' shuffles the
    same values.

    Because K is symmetric, W is reversible with stationary distribution
    pi_i proportional to w_i * d_i, and W is similar to the symmetric matrix
    S = G K G with G = diag(sqrt(w / d)). Both facts are used below.

Environment input:
    Each source node gives a gamma share of its attention to a fixed-opinion
    environment node and keeps the rest of its row in proportion. The time
    to align every node with it is ln(delta) / ln(rho), where rho is the
    spectral radius of the real-node block Q = diag(f) W (f = 1 - gamma on
    sources, 1 elsewhere). Q is similar to the symmetric diag(sqrt f) S
    diag(sqrt f), so rho comes from a symmetric eigensolve.
"""
import os
import sys

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh
from scipy.special import ndtri

PROBE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # probe/
REPO = os.path.dirname(PROBE)                                           # repo root: engine.py, generator.py
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from engine import trophic_coherence                      # noqa: E402  (unchanged)
from generator import largest_connected_component         # noqa: E402  (unchanged)

LN_DELTA = np.log(1e-6)   # same consensus threshold as engine.compute_spectral_gap


# --- network ---------------------------------------------------------------

def level_sizes(N, n_levels, r):
    '''Geometric pyramid: level l gets a share proportional to r**(l-1); largest-remainder rounding.'''
    raw = r ** np.arange(n_levels)
    raw = raw / raw.sum() * N
    k = np.floor(raw).astype(int)
    k[np.argsort(raw - k)[::-1][:N - k.sum()]] += 1
    if k.min() < 1:
        raise ValueError(f"N={N} too small for {n_levels} levels at r={r}")
    return k


def create_rule_b(N, n_levels, r, c, T, rng):
    '''Rule-B pyramid graph; edges point authority -> subordinate, nodes carry a 'level' attribute.'''
    k = level_sizes(N, n_levels, r)
    lv = np.repeat(np.arange(n_levels), k)               # 0-based level of each node id
    start = np.r_[0, np.cumsum(k)]
    L = np.arange(n_levels)
    log_kernel = -((L[:, None] - L[None, :] - 1) ** 2) / (2 * T ** 2)   # [source level, target level]

    src, dst = [], []
    for i in range(start[1], N):
        s = lv[i]
        avail = k.astype(float)
        avail[s] -= 1                                    # no self-loops
        chosen = set()
        for _ in range(c):
            # level chosen with prob. proportional to kernel x available nodes; working in logs
            # keeps the best available level at weight 1 even when the kernel underflows
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


def prepare(G):
    '''LCC, trophic levels, q, and the symmetric tie matrix K (undirected adjacency + I).'''
    Gc, n_dropped = largest_connected_component(G)
    tc = trophic_coherence(Gc)
    n = Gc.number_of_nodes()
    e = np.array(Gc.edges(), dtype=int).reshape(-1, 2)
    A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    K = (((A + A.T) > 0).astype(float) + identity(n, format='csr')).tocsr()
    return dict(G=Gc, n_dropped=n_dropped, s=tc['trophic_levels'],
                q=tc['trophic_incoherence'], K=K, n=n)


# --- influence weights and dynamics ---------------------------------------

def influence_weights(s, sigma, placement, rng):
    '''Lognormal quantiles (median 1), sorted by trophic level or shuffled.'''
    n = len(s)
    w_sorted = np.exp(sigma * ndtri((np.arange(1, n + 1) - 0.5) / n))
    if placement == 'sorted':
        order = np.lexsort((rng.random(n), s))           # ascending s, random tie-breaks
        w = np.empty(n)
        w[order] = w_sorted
        return w
    if placement == 'random':
        return rng.permutation(w_sorted)
    raise ValueError(placement)


def weight_matrix(K, w):
    '''Row-stochastic W = D^-1 K diag(w) and the row totals d = K w.'''
    d = K @ w
    return csr_matrix(diags(1 / d) @ K @ diags(w)), d


def internal_consensus(K, w, d):
    '''Internal consensus time (no environment): ln(delta)/ln|lambda_2| of W, via its symmetric twin.'''
    g = np.sqrt(w / d)
    S = (diags(g) @ K @ diags(g)).tocsr()
    vals = eigsh(S, k=2, which='LM', return_eigenvectors=False)
    lam2 = np.sort(np.abs(vals))[0]                      # the other one is the eigenvalue 1
    return LN_DELTA / np.log(lam2), lam2


def alignment(K, w, d, sources, gamma):
    '''Time to align with the environment, its rate 1-rho, and the influence-only estimate of the rate.'''
    f = np.ones(len(w))
    f[sources] = 1 - gamma
    g = np.sqrt(f * w / d)
    H = (diags(g) @ K @ diags(g)).tocsr()
    rho = eigsh(H, k=1, which='LA', return_eigenvectors=False)[0]
    pi = w * d / (w * d).sum()
    infl = pi[sources].sum()
    return LN_DELTA / np.log(rho), 1 - rho, gamma * infl, infl


def pick_sources(rule, s, frac, rng):
    '''Top / bottom fraction by trophic level (random tie-breaks) or a uniformly random fraction.'''
    n = len(s)
    k = max(1, round(frac * n))
    if rule == 'random':
        return rng.choice(n, size=k, replace=False)
    sign = -1.0 if rule == 'top' else 1.0
    return np.lexsort((rng.random(n), sign * s))[:k]


# --- q calibration ---------------------------------------------------------

def invert_q_table(q_targets, T_grid, q_med):
    '''T for each target q from a table of median q by T; NaN where the target exceeds the table.'''
    q_tab = np.maximum.accumulate(np.asarray(q_med, float))
    keep = np.r_[q_tab[1:] > q_tab[:-1], True]           # strictly increasing for np.interp
    q_tab, T_tab = q_tab[keep], np.asarray(T_grid, float)[keep]
    q_targets = np.asarray(q_targets, float)
    T = np.interp(q_targets, q_tab, T_tab)
    T[q_targets > q_tab[-1]] = np.nan
    return T
