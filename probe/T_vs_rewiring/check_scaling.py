"""
Checks for scaling_model.py, plus a timing estimate.

  1. Depth schedule and level sizes under the top-size rule.
  2. The shared endpoints: p = 0 vs T = 0.1, and p = 1 vs T = inf, give the same tie statistics
     and q (different seeds, so close, not identical).
  3. Continuous-time internal consensus matches a dense eigensolve of W.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

import scaling_model as sm                                 # noqa: E402
from scaling_model import am, mdl, build_weight_matrix     # noqa: E402

print("1. depth (top-size rule, r = 0.65, top >= 15):")
for N in [300, 1000, 3000, 10_000, 30_000, 100_000]:
    k = sm.level_counts(N)
    print(f"  N={N:<7} {len(k):2d} levels, top {k[-1]}, bottom {k[0]} ({k[0] / N:.1%})")

print("\n2. shared endpoints (N = 3,000, 3 graphs each):")
rng = np.random.default_rng(5)
for fam, par in [('T', 0.1), ('p', 0.0), ('T', np.inf), ('p', 1.0)]:
    rows = []
    for _ in range(3):
        P = mdl.prepare(sm.make_graph(fam, par, 3000, rng))
        st = sm.tie_stats(P['G'])
        rows.append([P['q'], st['frac_one_down'], st['frac_not_down'], st['mean_gap_dev'], P['n_dropped']])
    m = np.mean(rows, axis=0)
    print(f"  {fam} = {par:<4}: q {m[0]:.3f}  one-down {m[1]:.3f}  not-down {m[2]:.3f}  "
          f"mean |gap-1| {m[3]:.3f}  dropped {m[4]:.0f}")

print("\n3. continuous time vs dense eigensolve (N = 300):")
for fam, par in [('T', 0.1), ('T', 1.0), ('p', 0.1)]:
    P = mdl.prepare(sm.make_graph(fam, par, 300, rng))
    am.tag_asymmetric(P['G'])
    for alpha in [1.0, 6.0]:
        t, lam = sm.continuous_time(P['G'], alpha)
        ev = np.linalg.eigvals(build_weight_matrix(P['G'], alpha).toarray())
        ev = ev[np.argsort(-ev.real)]
        print(f"  {fam} = {par:<4} alpha {alpha}: {t:8.3f} vs dense {sm.LN_1E6 / (1 - ev[1].real):8.3f}")

print("\ntiming, N = 10,000:")
for fam, par in [('T', 1.0), ('p', 0.3)]:
    t0 = time.time()
    P = mdl.prepare(sm.make_graph(fam, par, 10_000, rng))
    am.tag_asymmetric(P['G'])
    t1 = time.time()
    for alpha in [1.0, 2.0, 3.0, 4.0, 6.0]:
        sm.continuous_time(P['G'], alpha)
    print(f"  {fam} = {par}: graph {t1 - t0:.1f}s, 5 solves {time.time() - t1:.1f}s")
