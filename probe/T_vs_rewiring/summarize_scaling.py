"""
Tables for results/2026-11-04_internal_T_vs_rewiring/SUMMARY.md, written to summary_tables.md.

For each family (T or p), parameter value and alpha:
  - median continuous-time internal consensus time at N = 300, 10,000 and 100,000
  - effective exponent: OLS slope of log(time) on log(N) over every graph, all N, and N >= 10,000
  - cost per added level: OLS slope of time on depth over every graph (depth 5 -> 19)
  - ratio to the flat network (T = inf) at N = 100,000
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'results', '2026-11-04_internal_T_vs_rewiring')
df = pd.read_csv(os.path.join(OUT, 'scaling.csv'))
sys.stdout = open(os.path.join(OUT, 'summary_tables.md'), 'w', encoding='utf-8')


def slope(g, x, y, log):
    xs = np.log(g[x]) if log else g[x]
    ys = np.log(g[y]) if log else g[y]
    return np.polyfit(xs, ys, 1)[0]


print(f"rows {len(df)}, errors {(df.error.fillna('') != '').sum()}, "
      f"max dropped share {df.n_dropped.div(df.N).max():.3f}, max |Im lambda_+| {df.lamP_im.abs().max():.1e}\n")

print("## Network descriptors (median over graphs, N = 10,000)\n")
print("| family | value | q | ties one level down | ties sideways or up | mean \\|gap − 1\\| |")
print("|---|---|---|---|---|---|")
d = df[(df.alpha == 1) & (df.N == 10_000)]
for (fam, pi), g in d.groupby(['family', 'param_index']):
    print(f"| {fam} | {g.param.iloc[0]:g} | {g.q.median():.3f} | {g.frac_one_down.median():.3f} | "
          f"{g.frac_not_down.median():.3f} | {g.mean_gap_dev.median():.2f} |")

flat = df[(df.family == 'T') & np.isinf(df.param)].groupby(['alpha', 'N']).t_cont.median()
for a in sorted(df.alpha.unique()):
    print(f"\n## α = {a:g}\n")
    print("| family | value | t(300) | t(10,000) | t(100,000) | exponent, all N | exponent, N ≥ 10k | "
          "steps per added level | ÷ flat at 100,000 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (fam, pi), g in df[df.alpha == a].groupby(['family', 'param_index']):
        m = g.groupby('N').t_cont.median()
        print(f"| {fam} | {g.param.iloc[0]:g} | {m[300]:.1f} | {m[10_000]:.1f} | {m[100_000]:.1f} | "
              f"{slope(g, 'N', 't_cont', True):+.3f} | {slope(g[g.N >= 10_000], 'N', 't_cont', True):+.3f} | "
              f"{slope(g, 'depth', 't_cont', False):.2f} | {m[100_000] / flat[(a, 100_000)]:.2f} |")
