"""
Checks for sweep_model.py, plus a timing estimate at N = 10,000.

  1. Self-tie 1 through the symmetric route equals the dense W built directly from its definition.
  2. The alpha-rule influence vector is stationary and nonnegative, and alignment time matches
     the influence-only estimate ln(delta) / ln(1 - gamma pi(S)).
  3. Every structure builds at every N, with the depth and level sizes intended.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

import sweep_model as sm                                   # noqa: E402
from sweep_model import am, mdl, build_weight_matrix       # noqa: E402

rng = np.random.default_rng(1)

print("1. self-tie 1, symmetric route vs dense W (N = 300, T = 0.5):")
P = mdl.prepare(sm.make_graph('fixed_r0.8', 300, 0.5, rng))
A = P['K'].toarray() - np.eye(P['n'])
S = mdl.pick_sources('bottom', P['s'], 0.1, rng)
for sigma, placement in [(1.0, 'sorted'), (2.0, 'random')]:
    w = mdl.influence_weights(P['s'], sigma, placement, rng)
    K1, d = sm.tie_matrix(P['K'], w, 'one')
    t_int, a_sym = mdl.internal_consensus(K1, w, d)[0], mdl.alignment(K1, w, d, S, 0.01)[0]
    raw = A * w[None, :] + np.eye(P['n'])                  # raw weight i places on j: w_j, self 1
    W = raw / raw.sum(1, keepdims=True)
    lam = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]
    f = np.ones(P['n'])
    f[S] = 0.99
    rho = np.abs(np.linalg.eigvals(f[:, None] * W)).max()
    print(f"  sigma {sigma} {placement:<6}  internal {t_int:9.2f} vs {mdl.LN_DELTA / np.log(lam[1]):9.2f}"
          f"   alignment {a_sym:11.1f} vs {mdl.LN_DELTA / np.log(rho):11.1f}"
          f"   max self-share {np.diag(W).max():.3f}")

print("\n2. alpha-rule influence (N = 1,000, grow_r0.5, T = 0.3):")
P = mdl.prepare(sm.make_graph('grow_r0.5', 1000, 0.3, rng))
am.tag_asymmetric(P['G'])
for alpha in [1.0, 2.0, 6.0]:
    pi = sm.influence_alpha(P['G'], alpha)
    W = build_weight_matrix(P['G'], alpha)
    for rule in ['top', 'bottom']:
        S = mdl.pick_sources(rule, P['s'], 0.1, rng)
        t = am.alignment_alpha(P['G'], alpha, S, 0.01, rng)
        est = mdl.LN_DELTA / np.log(1 - 0.01 * pi[S].sum())
        print(f"  alpha {alpha:<4} {rule:<6} residual {np.abs(pi @ W - pi).max():.1e}  min pi {pi.min():.1e}"
              f"   alignment {t:10.1f} vs estimate {est:10.1f}")

print("\n3. structures (levels: sizes, top first):")
for st in sm.STRUCTURES:
    for N in [300, 10_000]:
        k = mdl.level_sizes(N, sm.n_levels(st, N), sm.STRUCTURES[st][0])
        print(f"  {st:<11} N={N:<6} {len(k):2d} levels: {k[::-1].tolist()}")

print("\ntiming, N = 10,000, grow_r0.5 at T = 0.3:")
t0 = time.time()
P = mdl.prepare(sm.make_graph('grow_r0.5', 10_000, 0.3, rng))
am.tag_asymmetric(P['G'])
print(f"  graph {time.time() - t0:.1f}s, dropped {P['n_dropped']}")
S = mdl.pick_sources('bottom', P['s'], 0.1, rng)
for alpha in [1.0, 6.0]:
    t0 = time.time()
    am.internal_alpha(P['G'], alpha, rng)
    t1 = time.time()
    am.alignment_alpha(P['G'], alpha, S, 0.01, rng)
    t2 = time.time()
    sm.influence_alpha(P['G'], alpha)
    print(f"  alpha {alpha}: internal {t1 - t0:.1f}s  alignment {t2 - t1:.1f}s  influence {time.time() - t2:.1f}s")
