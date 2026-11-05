"""
Internal consensus time under three update rules, from the eigenvalues already in slow_mode.csv
(no new simulation), written to update_rule_tables.md in the results folder.

  synchronous   x(t+1) = W x(t)               time ln(delta) / ln|lambda_2|        (earlier runs)
  continuous    dx/dt = (W - I) x             time ln(1/delta) / (1 - Re lambda_+)
  half-weight   x(t+1) = (I + W)/2 x(t)       time ln(delta) / ln((1 + lambda_+)/2)

lambda_+ is W's eigenvalue with the largest real part after 1. In continuous time every mode
decays at rate 1 - Re(lambda), so negative eigenvalues decay fastest and can't set the time; the
half-weight rule maps lambda to (1 + lambda)/2, so they become small. (For the half-weight rule
this assumes no complex eigenvalue of modulus near 1, which the k = 6 solves show doesn't occur.)

Also compares two descriptions of the N-dependence, fitted separately on N <= 10,000 and N >= 10,000:
a power law (log time vs log N, exponent) and a log law (time vs ln N, slope in steps per unit
ln N). The description whose fitted parameter is stable across the two windows fits better.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'results', '2026-11-04_alpha_internal_slow_mode')
df = pd.read_csv(os.path.join(OUT, 'slow_mode.csv'))
LN = np.log(1e6)
df['t_sync'] = df.internal_time
df['t_cont'] = LN / (1 - df.lamP_re)
df['t_half'] = -LN / np.log((1 + df.lamP_re) / 2)
RULES = {'t_sync': 'synchronous', 't_cont': 'continuous', 't_half': 'half-weight'}
sys.stdout = open(os.path.join(OUT, 'update_rule_tables.md'), 'w', encoding='utf-8')
print(f"max |Im lambda_+| = {df.lamP_im.abs().max():.1e}\n")


def fit(g, col, lo, hi, kind):
    h = g[(g.N >= lo) & (g.N <= hi)]
    y = np.log(h[col]) if kind == 'power' else h[col]
    return np.polyfit(np.log(h.N), y, 1)[0]


print("## Hierarchy ÷ non-hierarchy, by update rule\n")
print("| structure | α | rule | N = 300 | 10,000 | 100,000 |")
print("|---|---|---|---|---|---|")
for st in ['grow_r0.8', 'grow_r0.65', 'fixed_r0.8']:
    for a in [1.0, 4.0, 6.0]:
        for col, name in RULES.items():
            m = df[(df.structure == st) & (df.alpha == a)].groupby(['q_target', 'N'])[col].median()
            r = m[0.05] / m[1.0]
            print(f"| {st} | {a:g} | {name} | {r[300]:.2f} | {r[10_000]:.2f} | {r[100_000]:.2f} |")

print("\n## Power law vs log law (growing structures; window N ≤ 10k → window N ≥ 10k)\n")
print("| structure | α | q | rule | power-law exponent | log-law slope (steps per unit ln N) |")
print("|---|---|---|---|---|---|")
for st in ['grow_r0.8', 'grow_r0.65']:
    for a in [1.0, 4.0, 6.0]:
        for qt in [0.05, 1.0]:
            g = df[(df.structure == st) & (df.alpha == a) & (df.q_target == qt)]
            for col, name in RULES.items():
                p1, p2 = fit(g, col, 300, 10_000, 'power'), fit(g, col, 10_000, 100_000, 'power')
                l1, l2 = fit(g, col, 300, 10_000, 'log'), fit(g, col, 10_000, 100_000, 'log')
                print(f"| {st} | {a:g} | {qt} | {name} | {p1:+.3f} → {p2:+.3f} | {l1:.1f} → {l2:.1f} |")
