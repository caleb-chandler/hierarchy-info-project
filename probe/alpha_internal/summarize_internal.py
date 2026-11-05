"""
Tables for results/2026-11-04_alpha_internal_slow_mode/SUMMARY.md, written to summary_tables.md.

'internal' is the usual internal consensus time, set by the largest |eigenvalue| after 1 (which
can be negative: an oscillation). 'non-osc' is the time set by the slowest non-oscillating mode
(largest real eigenvalue after 1). Exponents are OLS slopes of log(time) on log(N) over every
graph, with 95% CIs from the fit.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'results', '2026-11-04_alpha_internal_slow_mode')
df = pd.read_csv(os.path.join(OUT, 'slow_mode.csv'))
sys.stdout = open(os.path.join(OUT, 'summary_tables.md'), 'w', encoding='utf-8')
NS = sorted(df.N.unique())


def slope(g, col, lo, hi):
    h = g[(g.N >= lo) & (g.N <= hi)]
    (b, _), cov = np.polyfit(np.log(h.N), np.log(h[col]), 1, cov=True)
    return f"{b:+.2f} ± {1.96 * np.sqrt(cov[0, 0]):.2f}"


print(f"rows {len(df)}, errors {(df.error.fillna('') != '').sum()}, "
      f"max dropped share {df.n_dropped.div(df.N).max():.3f}\n")
print("## Measured q (median, [min, max]) by N\n")
print("| structure | target | " + " | ".join(f"N = {n:,}" for n in NS) + " |")
print("|---" * (2 + len(NS)) + "|")
for (st, qt), g in df[df.alpha == 1].groupby(['structure', 'q_target']):
    cells = [f"{h.q.median():.3f} [{h.q.min():.3f}, {h.q.max():.3f}]" for _, h in g.groupby('N')]
    print(f"| {st} | {qt} | " + " | ".join(cells) + " |")

print("\n## 1. Where the slowest mode sits (N = 10,000 and 100,000, medians)\n")
print("| structure | q | α | N | λ₂ | share of graphs with λ₂ < 0 | internal | non-osc | "
      "λ₂ mode: bottom / top-2 share | non-osc mode: bottom / top-2 share |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (st, qt, a, N), g in df[df.N.isin([10_000, 100_000])].groupby(['structure', 'q_target', 'alpha', 'N']):
    print(f"| {st} | {qt} | {a:g} | {N:,} | {g.lam2_re.median():+.3f} | {(g.lam2_re < 0).mean():.0%} | "
          f"{g.internal_time.median():.1f} | {g.nonosc_time.median():.1f} | "
          f"{g.bottom_share.median():.2f} / {g.top2_share.median():.2f} | "
          f"{g.nonosc_bottom_share.median():.2f} / {g.nonosc_top2_share.median():.2f} |")

for col, label in [('internal_time', 'internal (largest |λ|)'), ('nonosc_time', 'non-oscillating')]:
    print(f"\n## 2. Hierarchy vs non-hierarchy by N: {label}\n")
    for st in ['grow_r0.8', 'grow_r0.65', 'fixed_r0.8']:
        print(f"### {st}\n")
        print("| α | q | " + " | ".join(f"N = {n:,}" for n in NS) + " | exponent N ≤ 10k | exponent N ≥ 10k |")
        print("|---" * (4 + len(NS)) + "|")
        for a in [1.0, 2.0, 4.0, 6.0]:
            meds = {}
            for qt in [0.05, 1.0]:
                g = df[(df.structure == st) & (df.alpha == a) & (df.q_target == qt)]
                meds[qt] = g.groupby('N')[col].median()
                print(f"| {a:g} | {qt} | " + " | ".join(f"{meds[qt][n]:.1f}" for n in NS) +
                      f" | {slope(g, col, 300, 10_000)} | {slope(g, col, 10_000, 100_000)} |")
            ratio = meds[0.05] / meds[1.0]
            print(f"| {a:g} | ratio | " + " | ".join(f"**{ratio[n]:.2f}**" for n in NS) + " | | |")
        print()
