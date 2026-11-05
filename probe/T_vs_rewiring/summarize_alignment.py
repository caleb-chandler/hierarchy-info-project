"""
Tables for results/2026-11-04_alignment_T_vs_rewiring/SUMMARY.md, written to summary_tables.md.

Pairing check against scaling.csv (same graphs: q and N_actual must match). For each seeding,
family, parameter value and alpha: median alignment time at N = 300, 10,000 and 100,000, and the
scaling exponent (OLS slope of log time on log N over every graph, 95% CI). For the pure hierarchy,
bottom-seeded, the alpha * r prediction: the bottom level's influence falls by alpha * r per added
level and the top-size rule adds 1 / ln(1/r) levels per unit of ln N, so the exponent tends to
ln(alpha r) / ln(1/r) when alpha r > 1 (0 otherwise) as depth grows.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)
OUT = os.path.join(PROBE, 'results', '2026-11-04_alignment_T_vs_rewiring')
df = pd.read_csv(os.path.join(OUT, 'alignment.csv'))
sc = pd.read_csv(os.path.join(PROBE, 'results', '2026-11-04_internal_T_vs_rewiring', 'scaling.csv'))
sys.stdout = open(os.path.join(OUT, 'summary_tables.md'), 'w', encoding='utf-8')
R = 0.65

key = ['family', 'param_index', 'N', 'rep']
m = df[df.alpha == 1].drop_duplicates(key).merge(sc[sc.alpha == 1].drop_duplicates(key), on=key, suffixes=('', '_sc'))
print(f"rows {len(df)}, errors {(df.error.fillna('') != '').sum()}, unconverged influence rows {(~df.converged).sum()}")
print(f"pairing with scaling.csv: {len(m)} graphs matched; q identical {np.allclose(m.q, m.q_sc)}, "
      f"N_actual identical {(m.N_actual == m.N_actual_sc).all()}")
print(f"rows using the estimate (1 - rho <= 1e-9): {(df.method == 'estimate').mean():.1%}; "
      f"where both are valid, estimate ÷ exact time: median {(df.t_est / df.t_exact)[df.method == 'exact'].median():.3f}, "
      f"range {(df.t_est / df.t_exact)[df.method == 'exact'].min():.3f}–{(df.t_est / df.t_exact)[df.method == 'exact'].max():.3f}\n")


def slope(g):
    x, y = np.log(g.N), np.log(g.t_align)
    (b, _), cov = np.polyfit(x, y, 1, cov=True)
    return b, 1.96 * np.sqrt(cov[0, 0])


def fmt_t(t):
    return f"{t:,.0f}" if t < 1e6 else f"{t:.2g}"


for rule in ['top', 'bottom', 'random']:
    print(f"\n# {rule}-seeded alignment\n")
    for a in sorted(df.alpha.unique()):
        print(f"## α = {a:g}" + (f"  (α·r = {a * R:.2f}; hierarchy prediction {max(np.log(a * R), 0) / np.log(1 / R):.2f})"
                                 if rule == 'bottom' else '') + "\n")
        print("| family | value | t(300) | t(10,000) | t(100,000) | exponent [95% CI] | share estimated |")
        print("|---|---|---|---|---|---|---|")
        for (fam, pi), g in df[(df.seed_rule == rule) & (df.alpha == a)].groupby(['family', 'param_index']):
            med = g.groupby('N').t_align.median()
            b, ci = slope(g)
            print(f"| {fam} | {g.param.iloc[0]:g} | {fmt_t(med[300])} | {fmt_t(med[10_000])} | {fmt_t(med[100_000])} | "
                  f"{b:+.3f} ± {ci:.3f} | {(g.method == 'estimate').mean():.0%} |")
        print()
