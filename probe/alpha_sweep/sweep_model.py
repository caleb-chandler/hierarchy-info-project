"""
Model pieces for the 2026-11-04 runs: an alpha sweep, a check of the alpha * r threshold on
growing pyramids, and lognormal influence weights with the self-tie held at 1. Self-contained:
ablation/ablation_model.py, ruleB_overnight/model.py and engine.py are imported unchanged.

Structures (all rule B: every non-basal node draws exactly c = 6 subordinates):
    'fixed_r0.8'   8 levels, r = 0.8 (the overnight / ablation "new" structure)
    'grow_r0.8'    r = 0.8, 8/9/10/11 levels at N = 300/1,000/3,000/10,000 (the ablation's pyramid_grow)
    'grow_r0.65'   r = 0.65, same depths
    'grow_r0.5'    r = 0.5, 7/8/9/10 levels (one fewer, so the top level keeps 2+ nodes at N = 300)
  Every growing structure adds 3 levels over N = 300-10,000, so all add levels at the same rate.

Alpha rule: engine.build_weight_matrix (a junior weights its senior at alpha, the senior weights
the junior at 1, every self-tie is 1).

Threshold prediction: at q ~ 0 each level's members are about alpha times as influential as the
level below's and the level holds r times as many nodes, so level l holds influence ~ (alpha r)^l.
If alpha r > 1 the bottom level's share falls by alpha r per added level; if alpha r < 1 it
settles to a constant, so bottom-seeded alignment stops growing with N.

Lognormal weights with self-tie 1: W = D^-1 (A diag(w) + I), d = A w + 1. This is
D^-1 K1 diag(w) with K1 = A + diag(1/w), which is symmetric, so model.py's symmetric routes
(internal_consensus, alignment, pi ~ w d) apply unchanged with K1 in place of K = A + I.
"""
import os
import sys

import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import eigs

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)
if os.path.join(PROBE, 'ablation') not in sys.path:
    sys.path.insert(0, os.path.join(PROBE, 'ablation'))
import ablation_model as am                                # noqa: E402  (unchanged)
from ablation_model import mdl, build_weight_matrix        # noqa: E402  (unchanged)

C = 6
GROW_DEPTH = {0.8: {300: 8, 1000: 9, 3000: 10, 10_000: 11},
              0.65: {300: 8, 1000: 9, 3000: 10, 10_000: 11},
              0.5: {300: 7, 1000: 8, 3000: 9, 10_000: 10}}
STRUCTURES = {'fixed_r0.8': (0.8, False), 'grow_r0.8': (0.8, True),
              'grow_r0.65': (0.65, True), 'grow_r0.5': (0.5, True)}
ALPHAS = [1.0, 1.25, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0]
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]


def n_levels(structure, N):
    r, grow = STRUCTURES[structure]
    return GROW_DEPTH[r][N] if grow else 8


def make_graph(structure, N, T, rng):
    r = STRUCTURES[structure][0]
    return am.rule_b_on_levels(mdl.level_sizes(N, n_levels(structure, N), r), C, T, rng)


# --- alpha rule -------------------------------------------------------------

def influence_alpha(Gc, alpha):
    '''Stationary influence pi (pi W = pi, summing to 1) of the alpha-rule W.'''
    W = build_weight_matrix(Gc, alpha)
    n = W.shape[0]
    _, vecs = eigs(W.T.tocsr(), k=1, which='LM', ncv=min(n - 1, 30), tol=1e-12, maxiter=100_000,
                   v0=np.ones(n))
    pi = vecs[:, 0].real
    return pi / pi.sum()


# --- lognormal weights, self-tie rule ---------------------------------------

def tie_matrix(K, w, self_rule):
    '''Symmetric tie matrix for W = D^-1 K' diag(w): self-tie w_i ('own', the overnight rule)
    or 1 ('one'). Returns K' and the row totals d = K' w.'''
    if self_rule == 'own':
        return K, K @ w
    K1 = (K - identity(K.shape[0], format='csr') + diags(1 / w)).tocsr()
    return K1, K1 @ w
