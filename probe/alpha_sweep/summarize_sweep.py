"""
Tables for results/2026-11-04_alpha_sweep_selfloop/SUMMARY.md, written as markdown to
summary_tables.md in that folder.

Exponents are OLS slopes of log(time) on log(N) over every graph (N = 300-10,000), with 95% CIs
from the fit. 'influence-implied' is the same slope for 1 / pi(S), the weak-coupling estimate.
'level formula' is the slope of 1 / (bottom level's share) when level l holds influence
k_l alpha^l (k_l = level size), which is the alpha * r argument with no fitted inputs.
"""
import os
import sys

import numpy as np
import pandas as pd

import sweep_model as sm
from sweep_model import mdl

OUT = os.path.join(sm.PROBE, 'results', '2026-11-04_alpha_sweep_selfloop')
A = pd.read_csv(os.path.join(OUT, 'alpha.csv'))
L = pd.read_csv(os.path.join(OUT, 'lognormal.csv'))
N_RANGE = [300, 1000, 3000, 10_000]
sys.stdout = open(os.path.join(OUT, 'summary_tables.md'), 'w', encoding='utf-8')


def slope(df, col):
    x, y = np.log(df.N.values), np.log(df[col].values)
    (b, _), cov = np.polyfit(x, y, 1, cov=True)
    return b, 1.96 * np.sqrt(cov[0, 0])


def fmt(df, col):
    b, ci = slope(df, col)
    return f"{b:+.2f} ± {ci:.2f}"


def level_formula(structure, alpha):
    r = sm.STRUCTURES[structure][0]
    share = []
    for N in N_RANGE:
        k = mdl.level_sizes(N, sm.n_levels(structure, N), r)
        m = k * alpha ** np.arange(len(k))
        share.append(m[0] / m.sum())
    return np.polyfit(np.log(N_RANGE), -np.log(share), 1)[0]


def med10k(df, col):
    return df[df.N == 10_000][col].median()


print(f"alpha rows {len(A)}, errors {(A.error.fillna('') != '').sum()}; "
      f"lognormal rows {len(L)}, errors {(L.error.fillna('') != '').sum()}")
print(f"dropped nodes (max share by structure): "
      + ", ".join(f"{s} {g.n_dropped.div(g.N).max():.3f}" for s, g in A.groupby('structure')))
print(f"measured q by target (median):\n{A.groupby(['structure', 'q_target']).q.median().unstack().round(3)}\n")

# --- 1. threshold: bottom-seeded exponent vs alpha r ---------------------------------------
print("## 1. Threshold: bottom-seeded alignment exponent vs N\n")
for st in ['grow_r0.8', 'grow_r0.65', 'grow_r0.5', 'fixed_r0.8']:
    qs = sorted(A[A.structure == st].q_target.unique())
    print(f"### {st}\n")
    print("| alpha | alpha r | level formula | " + " | ".join(
        f"q={q}: measured | q={q}: infl.-implied" for q in qs) + " |")
    print("|---" * (3 + 2 * len(qs)) + "|")
    for a in sm.ALPHAS:
        g = A[(A.structure == st) & (A.alpha == a)].copy()
        g['inv_infl_bottom'] = 1 / g.infl_bottom
        cells = []
        for q in qs:
            h = g[g.q_target == q]
            cells += [fmt(h, 'align_bottom'), fmt(h, 'inv_infl_bottom')]
        print(f"| {a:g} | {a * sm.STRUCTURES[st][0]:.2f} | {level_formula(st, a):+.2f} | "
              + " | ".join(cells) + " |")
    print()

# per-member influence ratio between adjacent levels (q = 0.05, N = 10,000, middle levels)
print("### per-member influence of level l+1 / level l (q = 0.05, N = 10,000, median over graphs and levels 2-5)\n")
print("| structure | " + " | ".join(f"alpha {a:g}" for a in sm.ALPHAS) + " |")
print("|---" * (1 + len(sm.ALPHAS)) + "|")
for st in sm.STRUCTURES:
    cells = []
    for a in sm.ALPHAS:
        g = A[(A.structure == st) & (A.alpha == a) & (A.q_target == 0.05) & (A.N == 10_000)]
        k = mdl.level_sizes(10_000, sm.n_levels(st, 10_000), sm.STRUCTURES[st][0])
        ratios = []
        for s in g.level_infl:
            per = np.array([float(x) for x in s.split(';')]) / k[:len(s.split(';'))]
            ratios += list(per[2:6] / per[1:5])
        cells.append(f"{np.median(ratios):.2f}")
    print(f"| {st} | " + " | ".join(cells) + " |")
print()

# --- 2. alpha sweep: top / bottom / random, hierarchy vs non-hierarchy ------------------------
print("## 2. Alpha sweep\n")
for st in ['fixed_r0.8', 'grow_r0.8']:
    print(f"### {st}: median time at N = 10,000 (q = 0.05 / q = 1) and hierarchy ÷ non-hierarchy\n")
    print("| alpha | top | top ratio | bottom | bottom ratio | random | internal | internal ratio |")
    print("|---|---|---|---|---|---|---|---|")
    for a in sm.ALPHAS:
        g = A[(A.structure == st) & (A.alpha == a)]
        lo, hi = g[g.q_target == 0.05], g[g.q_target == 1.0]
        cells = []
        for col in ['align_top', 'align_bottom', 'align_random', 'internal_time']:
            m0, m1 = med10k(lo, col), med10k(hi, col)
            cells.append(f"{m0:,.0f} / {m1:,.0f}")
            if col != 'align_random':
                cells.append(f"{m0 / m1:.2f}")
        print(f"| {a:g} | " + " | ".join(cells) + " |")
    print()
    print(f"### {st}: exponent vs N (q = 0.05 / q = 1)\n")
    print("| alpha | top | bottom | random | internal |")
    print("|---|---|---|---|---|")
    for a in sm.ALPHAS:
        g = A[(A.structure == st) & (A.alpha == a)]
        lo, hi = g[g.q_target == 0.05], g[g.q_target == 1.0]
        print(f"| {a:g} | " + " | ".join(f"{fmt(lo, c)} / {fmt(hi, c)}" for c in
                                       ['align_top', 'align_bottom', 'align_random', 'internal_time']) + " |")
    print()

# --- 3. lognormal weights: self-tie own vs 1 ------------------------------------------------
print("## 3. Lognormal weights, self-tie w_i ('own') vs 1 ('one')\n")
for st in ['fixed_r0.8', 'grow_r0.8']:
    print(f"### {st}: exponent vs N, range over q = 0.05 / 0.3 / 1 (own → one)\n")
    print("| sigma | placement | top | random | bottom | internal |")
    print("|---|---|---|---|---|---|")
    for (sigma, pl), g in L[L.structure == st].groupby(['sigma', 'placement'], sort=False):
        cells = []
        for col in ['align_top', 'align_random', 'align_bottom', 'internal_time']:
            rng_ = {}
            for sr in ['own', 'one']:
                e = [slope(h, col)[0] for _, h in g[g.self_rule == sr].groupby('q_target')]
                rng_[sr] = f"{min(e):+.2f}…{max(e):+.2f}"
            cells.append(f"{rng_['own']} → {rng_['one']}")
        print(f"| {sigma:g} | {pl} | " + " | ".join(cells) + " |")
    print()
    print(f"### {st}: median at N = 10,000, q = 0.05 (own → one)\n")
    print("| sigma | placement | top | random | bottom | internal | max self-share |")
    print("|---|---|---|---|---|---|---|")
    for (sigma, pl), g in L[(L.structure == st) & (L.q_target == 0.05)].groupby(['sigma', 'placement'], sort=False):
        cells = []
        for col in ['align_top', 'align_random', 'align_bottom', 'internal_time', 'max_self_share']:
            o, n = (med10k(g[g.self_rule == sr], col) for sr in ['own', 'one'])
            cells.append(f"{o:.3f} → {n:.3f}" if col == 'max_self_share' else f"{o:,.0f} → {n:,.0f}")
        print(f"| {sigma:g} | {pl} | " + " | ".join(cells) + " |")
    print()

# influence-only estimate accuracy, self-tie 1
L['est'] = mdl.LN_DELTA / np.log(1 - 0.01 * L.infl_random)
r = (L.align_random / L.est)
print("exact ÷ influence-only estimate (random seeding), median and share off by > 20%:")
print(L.assign(ratio=r).groupby(['self_rule', 'sigma', 'placement']).ratio.agg(
    median='median', off20=lambda x: (abs(x - 1) > 0.2).mean()).round(3).to_string())
