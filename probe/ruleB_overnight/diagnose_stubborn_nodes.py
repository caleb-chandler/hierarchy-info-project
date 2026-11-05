"""
Diagnostic behind the "mechanism check" table in SUMMARY.md (test 3).

Question: is the slow internal consensus under shuffled high-sigma weights caused by
stubborn heavy nodes? Because each node's self-tie is scaled by its own weight, a
heavy node surrounded by light neighbors puts most of its attention on itself.

For each placement / sigma / N (3 fresh graphs, T = 0.35 so q ~ 0.16) it prints:
  internal T          internal consensus time
  T from stubborn     ln(delta) / ln(max self-share), the time a single node with that
                      self-share would need on its own
  max self-share      largest W_jj = w_j / d_j
  slow-mode PR        participation ratio of the slowest mode (how many nodes carry it)
  weight on heaviest  share of the slow mode on the heaviest 1% of nodes

Usage: python diagnose_stubborn_nodes.py   (about a minute)
"""
import os
import sys

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as mdl   # noqa: E402

rng = np.random.default_rng(7)
print(f"{'placement':<9} {'sigma':>5} {'N':>6} | {'internal T':>10} {'T from stubborn':>15} | "
      f"{'max self-share':>14} {'slow-mode PR':>12} {'weight on heaviest 1%':>22}")
for placement in ['random', 'sorted']:
    for sigma in [2.0, 1.0]:
        for N in [300, 1000, 3000, 10000]:
            out = []
            for _ in range(3):
                P = mdl.prepare(mdl.create_rule_b(N, 8, 0.8, 6, 0.35, rng))
                K, s, n = P['K'], P['s'], P['n']
                w = mdl.influence_weights(s, sigma, placement, rng)
                d = K @ w
                self_share = w / d
                t_int, _ = mdl.internal_consensus(K, w, d)
                g = np.sqrt(w / d)
                vals, vecs = eigsh((diags(g) @ K @ diags(g)).tocsr(), k=2, which='LM')
                v = vecs[:, np.argsort(np.abs(vals))[0]] / g      # right eigenvector of W for lambda_2
                m = v ** 2 / (v ** 2).sum()
                heavy = np.argsort(-w)[:max(1, n // 100)]
                out.append((t_int, mdl.LN_DELTA / np.log(self_share.max()), self_share.max(),
                            1 / (m ** 2).sum(), m[heavy].sum()))
            a = np.median(np.array(out), axis=0)
            print(f"{placement:<9} {sigma:5.1f} {N:6d} | {a[0]:10.0f} {a[1]:15.0f} | "
                  f"{a[2]:14.4f} {a[3]:12.1f} {a[4]:22.2f}")
