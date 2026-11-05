"""
Why sorted sigma = 2 weights still give bottom-seeded alignment a size effect with the self-tie
at 1: the mean of w^2 over the lognormal quantile grid keeps growing over N = 300-10,000, because
the largest quantile does. Under sorting, heavy nodes are tied to heavy nodes, so the top's
influence (pi_i ~ w_i d_i) grows roughly like w^2. Also confirms the alignment exponent equals
the exponent of 1 / pi(bottom 10%).
"""
import os

import numpy as np
import pandas as pd
from scipy.special import ndtri

PROBE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Ns = np.array([300, 1000, 3000, 10_000])
for s in [1.5, 2.0]:
    m1 = [np.exp(s * ndtri((np.arange(1, n + 1) - 0.5) / n)).mean() for n in Ns]
    m2 = [np.exp(2 * s * ndtri((np.arange(1, n + 1) - 0.5) / n)).mean() for n in Ns]
    print(f"sigma {s}: mean w {np.round(m1, 2)} slope {np.polyfit(np.log(Ns), np.log(m1), 1)[0]:.3f}"
          f" | mean w^2 {np.round(m2, 0)} slope {np.polyfit(np.log(Ns), np.log(m2), 1)[0]:.3f}")

L = pd.read_csv(os.path.join(PROBE, 'results', '2026-11-04_alpha_sweep_selfloop', 'lognormal.csv'))
g = L[(L.structure == 'fixed_r0.8') & (L.placement == 'sorted') & (L.sigma == 2)]
for (sr, q), h in g.groupby(['self_rule', 'q_target']):
    b = np.polyfit(np.log(h.N), np.log(1 / h.infl_bottom), 1)[0]
    a = np.polyfit(np.log(h.N), np.log(h.align_bottom), 1)[0]
    print(f"self-tie {sr}, q = {q}: alignment exponent {a:.3f}, 1 / influence exponent {b:.3f}")
