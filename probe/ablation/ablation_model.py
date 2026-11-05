"""
Ablation of the three changes made together for the 2026-10-29 overnight run. Each one is
switched between its old and new form; everything else (trophic levels, q, largest connected
component, environment input) is shared with ruleB_overnight/model.py.

  levels   'tree'     old: levels come from the GPPM's uniform-attachment tree on b = 0.1 basal
                      nodes, so depth and level occupancy are random.
           'pyramid'  new: 8 levels, each 0.8 times the one below.
           split of the levels change (SPLIT_STRUCTURES):
           'tree_cap8'      the tree with no node above level 8 (fixed depth, tree-shaped random occupancy)
           'tree_cap8_b24'  same with 24% basal nodes, the pyramid's basal share
           'pyramid_grow'   pyramid occupancy with the uncapped tree's median depth (8/9/10/11 levels
                            at N = 300/1,000/3,000/10,000)
  edges    'gppm'     old: one tree subordinate per non-basal node, then kernel-weighted excess
                      edges up to L = c_old * N (c_old = calibrate_density(0.1, 10000, 9) = 7.12).
           'c6'       new: every non-basal node draws exactly 6 subordinates (rule B).
           'gppm_matched'  extra: GPPM edges with rule B's edge count (6 per non-basal node),
                      to tell the effect of regular degree from the effect of fewer edges.
  weights  'alpha2'   old: engine.build_weight_matrix with alpha = 2 (a junior weights its senior
                      at 2, the senior weights the junior at 1, every self-tie is 1).
           'sorted'   new: lognormal quantiles, sigma = 0.5, sorted by trophic level,
                      W = D^-1 K diag(w) (self-tie scaled by the node's own weight).
           'equal'    reference: alpha = 1, which is the same matrix as sigma = 0.

tree + gppm is generator.create_new and pyramid + c6 is model.create_rule_b, both unchanged.
The two mixed structures reuse their pieces: tree + c6 runs rule B on levels drawn by the
GPPM tree step; pyramid + gppm builds a one-subordinate tree on the pyramid's levels and adds
the GPPM's excess edges with the same bucket rule as create_new.
"""
import os
import sys

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)                              # probe/: sibling code, results, figures
REPO = os.path.dirname(PROBE)                              # repo root: engine.py, generator.py
for p in (REPO, os.path.join(PROBE, 'ruleB_overnight')):
    if p not in sys.path:
        sys.path.insert(0, p)
import generator as gen                                    # noqa: E402  (unchanged)
from engine import build_weight_matrix                     # noqa: E402  (unchanged)
import model as mdl                                        # noqa: E402  (overnight model)

B_FRAC = 0.1
C_OLD = gen.calibrate_density(B_FRAC, 10_000, margin=9.0)
D_NEW, R_NEW, C_NEW = 8, 0.8, 6
ALPHA_OLD, SIGMA_NEW = 2.0, 0.5
STRUCTURES = [('tree', 'gppm'), ('tree', 'c6'), ('pyramid', 'gppm'), ('pyramid', 'c6')]
EXTRA_STRUCTURES = [('pyramid', 'gppm_matched')]           # separates density from regular degree
# splitting the levels change into depth, occupancy shape and basal share
GROW_DEPTH = {300: 8, 1000: 9, 3000: 10, 10_000: 11}       # the uncapped tree's median depth
SPLIT_STRUCTURES = [(lv, ed) for lv in ['tree_cap8', 'tree_cap8_b24', 'pyramid_grow']
                    for ed in ['gppm', 'c6']]


# --- levels -----------------------------------------------------------------

def tree_attach(N, b, rng, cap=None):
    '''GPPM tree step: each new node takes a uniformly random existing node as its subordinate.
    With cap, nodes already at level cap can't be picked, so no node goes above it.
    Returns each node's level (creation order) and the tree edges (authority -> subordinate).'''
    B = max(1, round(b * N))
    lv = np.ones(N, dtype=int)
    eligible = list(range(B))
    edges = []
    for n in range(B, N):
        sub = eligible[rng.integers(len(eligible))]
        lv[n] = lv[sub] + 1
        edges.append((n, sub))
        if cap is None or lv[n] < cap:
            eligible.append(n)
    return lv, edges


def tree_levels(N, b, rng, cap=None):
    '''Node count per level (level 1 first) from the GPPM tree step.'''
    return np.bincount(tree_attach(N, b, rng, cap)[0])[1:]


# --- edges ------------------------------------------------------------------

def rule_b_on_levels(k, c, T, rng):
    '''Rule B (model.create_rule_b) on arbitrary level counts k; node ids are grouped by level.'''
    k = np.asarray(k)
    n_levels, N = len(k), int(k.sum())
    lv = np.repeat(np.arange(n_levels), k)
    start = np.r_[0, np.cumsum(k)]
    L = np.arange(n_levels)
    log_kernel = -((L[:, None] - L[None, :] - 1) ** 2) / (2 * T ** 2)

    src, dst = [], []
    for i in range(start[1], N):
        s = lv[i]
        avail = k.astype(float)
        avail[s] -= 1
        chosen = set()
        for _ in range(c):
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


def gppm_on_levels(levels, L_total, T, rng, tree_edges=None):
    '''create_new's excess-edge step on given per-node levels (1-based). Without tree_edges, each
    non-basal node first gets one subordinate drawn uniformly from the level below.'''
    levels = np.asarray(levels)
    N = len(levels)
    if tree_edges is None:
        by_level = {lvl: np.where(levels == lvl)[0] for lvl in np.unique(levels)}
        tree_edges = []
        for n in np.where(levels > 1)[0]:
            below = by_level[levels[n] - 1]
            tree_edges.append((int(n), int(below[rng.integers(len(below))])))
    tree_edge_set = set(tree_edges)

    # excess edges: same buckets, kernel and rejection rule as generator.create_new
    non_basal_ids = np.where(levels > 1)[0]
    sources_by_level = {lvl: non_basal_ids[levels[non_basal_ids] == lvl]
                        for lvl in np.unique(levels[non_basal_ids])}
    targets_by_level = {lvl: np.where(levels == lvl)[0] for lvl in np.unique(levels)}
    bucket_sl, bucket_tl, bucket_mass = [], [], []
    for sl, src_ids in sources_by_level.items():
        for tl, tgt_ids in targets_by_level.items():
            count = len(src_ids) * len(tgt_ids)
            if sl == tl:
                count -= len(tgt_ids)
            if tl == sl - 1:
                count -= len(src_ids)
            if count <= 0:
                continue
            bucket_sl.append(sl)
            bucket_tl.append(tl)
            bucket_mass.append(np.exp(-((sl - tl - 1) ** 2) / (2 * T ** 2)) * count)
    bucket_probs = np.array(bucket_mass) / np.sum(bucket_mass)

    n_excess = L_total - len(tree_edges)
    chosen = set()
    for _ in range(200):
        if len(chosen) >= n_excess:
            break
        n_draw = int((n_excess - len(chosen)) * 1.2) + 10
        for bi in rng.choice(len(bucket_probs), size=n_draw, p=bucket_probs):
            src_ids, tgt_ids = sources_by_level[bucket_sl[bi]], targets_by_level[bucket_tl[bi]]
            s = int(src_ids[rng.integers(len(src_ids))])
            t = int(tgt_ids[rng.integers(len(tgt_ids))])
            if s == t or (s, t) in tree_edge_set or (s, t) in chosen:
                continue
            chosen.add((s, t))
            if len(chosen) >= n_excess:
                break
    else:
        raise RuntimeError("could not sample enough excess edges")

    G = nx.DiGraph()
    G.add_nodes_from((n, {'level': int(levels[n])}) for n in range(N))
    G.add_edges_from(tree_edges)
    G.add_edges_from(chosen)
    return G


def make_graph(levels, edges, N, T, rng):
    if levels == 'tree' and edges == 'gppm':
        return gen.create_new(N, B_FRAC, C_OLD, T, rng)[0]
    if levels == 'pyramid' and edges == 'c6':
        return mdl.create_rule_b(N, D_NEW, R_NEW, C_NEW, T, rng)
    if levels.startswith('tree'):
        b = 0.24 if levels.endswith('_b24') else B_FRAC
        lv, tree_edges = tree_attach(N, b, rng, cap=8 if '_cap8' in levels else None)
        k = np.bincount(lv)[1:]
    else:
        k = mdl.level_sizes(N, GROW_DEPTH[N] if levels == 'pyramid_grow' else D_NEW, R_NEW)
        lv, tree_edges = np.repeat(np.arange(1, len(k) + 1), k), None
    if edges == 'c6':
        return rule_b_on_levels(k, C_NEW, T, rng)
    if edges == 'gppm_matched':                            # GPPM edges at rule B's edge count
        return gppm_on_levels(lv, C_NEW * int(N - k[0]), T, rng, tree_edges)
    return gppm_on_levels(lv, round(C_OLD * N), T, rng, tree_edges)


def tag_asymmetric(G):
    '''The edge attribute engine.build_weight_matrix needs (create_new sets it; the others don't).'''
    for u, v in G.edges():
        G[u][v]['is_asymmetric'] = not G.has_edge(v, u)


# --- alpha-rule times (W is not reversible once q > 0, so general eigensolves) ----

def internal_alpha(Gc, alpha, rng):
    W = build_weight_matrix(Gc, alpha)
    n = W.shape[0]
    vals = eigs(W, k=3, which='LM', ncv=min(n - 1, 40), tol=1e-10, maxiter=100_000,
                v0=rng.random(n), return_eigenvectors=False)
    mod = np.sort(np.abs(vals))[::-1]
    return mdl.LN_DELTA / np.log(mod[1]), mod[0]


def alignment_alpha(Gc, alpha, sources, gamma, rng):
    W = build_weight_matrix(Gc, alpha, sources=sources, gamma=gamma)
    n = Gc.number_of_nodes()
    Q = W[:n, :n]
    vals = eigs(Q, k=1, which='LM', ncv=min(n - 1, 30), tol=1e-12, maxiter=100_000,
                v0=np.ones(n), return_eigenvectors=False)
    return mdl.LN_DELTA / np.log(np.abs(vals[0]))
