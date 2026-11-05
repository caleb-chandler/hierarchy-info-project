"""
Checks for the alignment additions to scaling_model.py.

  1. Power-iteration influence matches the eigensolver's (alpha_sweep/sweep_model.influence_alpha)
     where the eigensolver is accurate.
  2. The weak-coupling estimate ln(1e6) / (gamma pi(S)) matches the exact ln(1e6) / (1 - rho),
     including bottom-seeded cases where pi(S) is ~1e-6 to 1e-8.
  3. Timing at N = 10,000 and 100,000.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys                                                 # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

import scaling_model as sm                                 # noqa: E402
from scaling_model import am, mdl                          # noqa: E402
sys.path.insert(0, os.path.join(sm.PROBE, 'alpha_sweep'))
from sweep_model import influence_alpha                    # noqa: E402

GAMMA = 0.01
rng = np.random.default_rng(11)
for fam, par, N in [('T', 0.1, 1000), ('T', 1.5, 1000), ('p', 0.03, 3000)]:
    P = mdl.prepare(sm.make_graph(fam, par, N, rng))
    am.tag_asymmetric(P['G'])
    S = {r: mdl.pick_sources(r, P['s'], 0.10, rng) for r in ['top', 'random', 'bottom']}
    print(f"{fam} = {par}, N = {N} ({sm.depth(N)} levels):")
    for alpha in [1.0, 2.0, 4.0, 6.0]:
        pi, it, ok = sm.influence(P['G'], alpha, list(S.values()))
        pi_e = influence_alpha(P['G'], alpha)
        big = pi_e > 1e-6 * pi_e.max()
        cells = []
        for r, src in S.items():
            gap = sm.alignment_gap(P['G'], alpha, src, GAMMA)
            cells.append(f"{r} pi(S) {pi[src].sum():.2e} est/exact {(1 - gap) and gap / (GAMMA * pi[src].sum()):.4f}")
        print(f"  alpha {alpha}: {it} iterations ({'converged' if ok else 'NOT converged'}), "
              f"max rel. diff vs eigensolver {np.max(np.abs(pi[big] / pi_e[big] - 1)):.1e}; " + "; ".join(cells))

print("\ntiming (influence + 3 alignment solves per alpha):")
for N in [10_000, 100_000]:
    P = mdl.prepare(sm.make_graph('T', 0.1, N, rng))
    am.tag_asymmetric(P['G'])
    S = {r: mdl.pick_sources(r, P['s'], 0.10, rng) for r in ['top', 'random', 'bottom']}
    for alpha in [1.0, 6.0]:
        t0 = time.time()
        pi, it, ok = sm.influence(P['G'], alpha, list(S.values()))
        t1 = time.time()
        for src in S.values():
            sm.alignment_gap(P['G'], alpha, src, GAMMA)
        print(f"  N = {N}, alpha {alpha}: influence {t1 - t0:.1f}s ({it} it., {ok}), 3 solves {time.time() - t1:.1f}s; "
              f"bottom pi(S) {pi[S['bottom']].sum():.2e}")
