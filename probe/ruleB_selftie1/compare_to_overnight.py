"""
Pairs every row of the self-tie-1 rerun with the same graph, weighting and seeding in the
2026-10-29 overnight run, and writes comparison_tables.md (overnight → self-tie 1) to the
rerun's results folder.

Pairing checks: q, N_actual and n_sources must be identical (same graphs and sources), and the
sigma = 0 rows must give the same times (with equal weights both self-tie rules are one matrix).
Exponents are OLS slopes of log(time) on log(N) over every graph in a cell, as in make_figures.py.
"""
import os
import sys

import numpy as np
import pandas as pd

PROBE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = os.path.join(PROBE, 'results', '2026-10-29_ruleB_pyramid_weights')
NEW = os.path.join(PROBE, 'results', '2026-11-04_ruleB_selftie1')
TESTS = ['test1_hierarchy_q0', 'test2_random_seeding', 'test3_weight_distribution', 'test4_depth']
KEY = ['D', 'N', 'q_index', 'rep', 'sigma', 'placement', 'seed_rule']
sys.stdout = open(os.path.join(NEW, 'comparison_tables.md'), 'w', encoding='utf-8')


def load(name):
    o = pd.read_csv(os.path.join(OLD, name + '.csv'))
    n = pd.read_csv(os.path.join(NEW, name + '.csv'))
    m = o.merge(n, on=KEY, suffixes=('_old', '_new'), validate='one_to_one')
    assert len(m) == len(o) == len(n), (name, len(o), len(n), len(m))
    for c in ['q', 'N_actual', 'n_sources']:
        assert np.allclose(m[c + '_old'], m[c + '_new']), (name, c)
    z = m[m.sigma == 0]
    dev = np.abs(z.align_time_new / z.align_time_old - 1).max() if len(z) else 0.0
    print(f"- {name}: {len(m)} paired rows; sigma = 0 max relative difference {dev:.1e}")
    return m


def slope(df, col):
    return np.polyfit(np.log(df.N), np.log(df[col]), 1)[0]


def rng_over_q(df, col):
    e = [slope(h, col) for _, h in df.groupby('q_index')]
    return f"{min(e):+.2f}…{max(e):+.2f}"


def both(df, f, col):
    return f"{f(df, col + '_old')} → {f(df, col + '_new')}"


def med(df, col):
    return f"{df[col].median():,.0f}"


print("## Pairing\n")
T1, T2, T3, T4 = (load(t) for t in TESTS)

# --- test 1 ---
print("\n## Test 1 · q = 0, N = 100–10,000\n")
print("| weights | seeding | time at N = 10,000 | exponent |")
print("|---|---|---|---|")
for sigma in [0.5, 0.0]:
    for rule in ['top', 'random', 'bottom']:
        g = T1[(T1.sigma == sigma) & (T1.seed_rule == rule)]
        g10 = g[g.N == 10_000]
        print(f"| sorted σ = {sigma:g} | {rule} | {both(g10, med, 'align_time')} | "
              f"{slope(g, 'align_time_old'):+.3f} → {slope(g, 'align_time_new'):+.3f} |")
g = T1[(T1.sigma == 0.5) & (T1.seed_rule == 'top')]
print(f"\nInternal consensus, sorted σ = 0.5: N = 100 {both(g[g.N == 100], med, 'internal_time')}; "
      f"N = 10,000 {both(g[g.N == 10_000], med, 'internal_time')}")

# --- test 2 ---
print("\n## Test 2 · sorted σ = 0.5, 10 q values 0.05–1.0\n")
print("| seeding | exponent range over q | time at N = 10,000, range over q |")
print("|---|---|---|")
for rule in ['random', 'top', 'bottom']:
    g = T2[T2.seed_rule == rule]
    t = g[g.N == 10_000].groupby('q_index')[['align_time_old', 'align_time_new']].median()
    print(f"| {rule} | {both(g, rng_over_q, 'align_time')} | "
          f"{t.align_time_old.min():,.0f}–{t.align_time_old.max():,.0f} → "
          f"{t.align_time_new.min():,.0f}–{t.align_time_new.max():,.0f} |")

# --- test 3 ---
print("\n## Test 3 · sorted vs shuffled, 8 q values 0.05–1.25\n")
print("Exponent of alignment time vs N, range over q (overnight → self-tie 1):\n")
conds = [(s, 'sorted') for s in [0.0, 0.5, 1.0, 1.5, 2.0]] + [(s, 'random') for s in [0.5, 1.0, 1.5, 2.0]]
print("| weights | random | top | bottom | internal |")
print("|---|---|---|---|---|")
for sigma, pl in conds:
    g = T3[(T3.sigma == sigma) & (T3.placement == pl)]
    cells = [both(g[g.seed_rule == r], rng_over_q, 'align_time') for r in ['random', 'top', 'bottom']]
    gi = g[g.seed_rule == 'random']
    cells.append(both(gi, rng_over_q, 'internal_time'))
    name = f"{'sorted' if pl == 'sorted' else 'shuffled'} σ = {sigma:g}"
    print(f"| {name} | " + " | ".join(cells) + " |")

print("\nMedian time at N = 10,000 over all q (overnight → self-tie 1):\n")
print("| weights | random | top | bottom | internal | max self-share |")
print("|---|---|---|---|---|---|")
for sigma, pl in conds:
    g = T3[(T3.sigma == sigma) & (T3.placement == pl) & (T3.N == 10_000)]
    cells = [both(g[g.seed_rule == r], med, 'align_time') for r in ['random', 'top', 'bottom']]
    cells.append(both(g[g.seed_rule == 'random'], med, 'internal_time'))
    cells.append(f"— → {g.max_self_share.median():.3f}")
    name = f"{'sorted' if pl == 'sorted' else 'shuffled'} σ = {sigma:g}"
    print(f"| {name} | " + " | ".join(cells) + " |")

print("\nSorted ÷ shuffled alignment time at N = 10,000, median over q (overnight → self-tie 1):\n")
print("| seeding | σ = 0.5 | σ = 1 | σ = 1.5 | σ = 2 |")
print("|---|---|---|---|---|")
for rule in ['top', 'random', 'bottom']:
    cells = []
    for sigma in [0.5, 1.0, 1.5, 2.0]:
        g = T3[(T3.sigma == sigma) & (T3.seed_rule == rule) & (T3.N == 10_000)]
        out = []
        for c in ['align_time_old', 'align_time_new']:
            m = g.groupby(['q_index', 'placement'])[c].median().unstack()
            out.append((m['sorted'] / m['random']).median())
        cells.append(f"{out[0]:.3g} → {out[1]:.3g}")
    print(f"| {rule} | " + " | ".join(cells) + " |")

print("\nInfluence-only estimate (γ × source influence) ÷ exact rate: median, and share of graphs off by > 20%:\n")
print("| weights | overnight | self-tie 1 |")
print("|---|---|---|")
for sigma, pl in conds:
    g = T3[(T3.sigma == sigma) & (T3.placement == pl)]
    cells = []
    for sfx in ['_old', '_new']:
        r = g['approx_rate' + sfx] / g['align_rate' + sfx]
        cells.append(f"{r.median():.3f}, {(np.abs(r - 1) > 0.2).mean():.0%}")
    name = f"{'sorted' if pl == 'sorted' else 'shuffled'} σ = {sigma:g}"
    print(f"| {name} | " + " | ".join(cells) + " |")

# --- test 4 ---
print("\n## Test 4 · depth 4–12, sorted σ = 0.5\n")
qi_lo, qi_hi = 0, 9
print("Alignment time at N = 10,000 (overnight → self-tie 1):\n")
print("| seeding, q | " + " | ".join(f"{d} levels" for d in [4, 6, 8, 10, 12]) + " |")
print("|---|---|---|---|---|---|")
for rule, qi, lab in [('top', qi_lo, 'top, q = 0.05'), ('top', qi_hi, 'top, q = 1'),
                      ('random', qi_lo, 'random, q = 0.05'), ('bottom', qi_lo, 'bottom, q = 0.05'),
                      ('bottom', qi_hi, 'bottom, q = 1')]:
    cells = []
    for d in [4, 6, 8, 10, 12]:
        g = T4[(T4.D == d) & (T4.seed_rule == rule) & (T4.q_index == qi) & (T4.N == 10_000)]
        cells.append(both(g, med, 'align_time'))
    print(f"| {lab} | " + " | ".join(cells) + " |")
print("\nInternal consensus at N = 10,000:\n")
print("| q | " + " | ".join(f"{d} levels" for d in [4, 6, 8, 10, 12]) + " |")
print("|---|---|---|---|---|---|")
for qi, lab in [(qi_lo, '0.05'), (qi_hi, '1.0')]:
    cells = []
    for d in [4, 6, 8, 10, 12]:
        g = T4[(T4.D == d) & (T4.seed_rule == 'top') & (T4.q_index == qi) & (T4.N == 10_000)]
        cells.append(both(g, med, 'internal_time'))
    print(f"| {lab} | " + " | ".join(cells) + " |")
print("\nAlignment exponent vs N, range over depths and q (overnight → self-tie 1):\n")
for rule in ['top', 'random', 'bottom']:
    g = T4[T4.seed_rule == rule]
    e = {sfx: [slope(h, 'align_time' + sfx) for _, h in g.groupby(['D', 'q_index'])] for sfx in ['_old', '_new']}
    print(f"- {rule}: {min(e['_old']):+.3f}…{max(e['_old']):+.3f} → {min(e['_new']):+.3f}…{max(e['_new']):+.3f}")
