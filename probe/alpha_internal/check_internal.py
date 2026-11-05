"""
Check for run_internal.py: the slow-mode solver (k = 6, with eigenvectors) gives the same internal
time as ablation_model.internal_alpha (k = 3), and the eigenvector satisfies W v = lambda v.
"""
import numpy as np

import run_internal as ri
from run_internal import am, mdl, build_weight_matrix

rng = np.random.default_rng(3)
for st, N, T in [('fixed_r0.8', 1000, 0.272), ('grow_r0.65', 3000, 1.709)]:
    D = ri.n_levels(st, N)
    P = mdl.prepare(am.rule_b_on_levels(mdl.level_sizes(N, D, ri.sm.STRUCTURES[st][0]), 6, T, rng))
    am.tag_asymmetric(P['G'])
    levels = np.array([P['G'].nodes[n]['level'] for n in range(P['n'])])
    for alpha in [1.0, 6.0]:
        lam2, lam3, share, part, lp, pshare, ppart = ri.slow_mode(P['G'], alpha, levels, D)
        t_ref = am.internal_alpha(P['G'], alpha, rng)[0]
        print(f"{st:<11} N={N} q={P['q']:.3f} alpha={alpha}: internal {mdl.LN_DELTA / np.log(abs(lam2)):7.2f}"
              f" vs {t_ref:7.2f}  lambda2 {lam2:.4f}  |lambda3| {lam3:.4f}  top-2 share {share[-2:].sum():.3f}"
              f"  | non-osc lambda {lp:.4f} time {mdl.LN_DELTA / np.log(abs(lp)):7.2f}"
              f" top-2 {pshare[-2:].sum():.3f} bottom {pshare[0]:.3f}")
        print("      oscillating-mode level shares (bottom first):", np.round(share, 3).tolist())
W = build_weight_matrix(P['G'], 6.0)
from scipy.sparse.linalg import eigs                       # noqa: E402
vals, vecs = eigs(W, k=2, which='LM')
print("residual |W v - lambda v|:", np.abs(W @ vecs[:, 1] - vals[1] * vecs[:, 1]).max())
