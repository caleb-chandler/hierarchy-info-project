"""
Tables for results/2026-11-05_internal_alpha10/: alpha = 10 next to alpha = 6 (from the 11-04 run) on
the same graphs. Pairing check: q and node count must match the 11-04 rows graph by graph.

  python summarize_alpha10.py    -> results/2026-11-05_internal_alpha10/summary_tables.md
"""
import os

import numpy as np
import pandas as pd

from fig_style import PROBE, slope_ci

RES = os.path.join(PROBE, 'results')
new = pd.read_csv(os.path.join(RES, '2026-11-05_internal_alpha10', 'scaling_alpha10.csv'))
old = pd.read_csv(os.path.join(RES, '2026-11-04_internal_T_vs_rewiring', 'scaling.csv'))
KEY = ['family', 'param_index', 'N', 'rep']

pair = new.merge(old[old.alpha == 6][KEY + ['q', 'N_actual']], on=KEY, suffixes=('', '_old'))
bad = (~np.isclose(pair.q, pair.q_old)) | (pair.N_actual != pair.N_actual_old)
lines = [f"Pairing: {len(pair)} of {len(new)} alpha = 10 graphs matched to 11-04 graphs, "
         f"{int(bad.sum())} mismatched (q or node count). Errors: {(new.error.fillna('') != '').sum()}.", '']

# compare on exactly the graphs that have alpha = 10 rows
both = pd.concat([old[old.alpha == 6].merge(new[KEY], on=KEY), new], ignore_index=True)
both = both[both.error.fillna('') == '']
flat = both[(both.family == 'T') & np.isinf(both.param) & (both.N == 100_000)].groupby('alpha').t_cont.median()

rows = []
for (fam, v, a), g in both.groupby(['family', 'param', 'alpha']):
    b, lo, hi = slope_ci(g, 't_cont')
    cost = np.polyfit(g.depth, g.t_cont, 1)[0]
    t100 = g[g.N == 100_000].t_cont.median()
    rows.append(dict(family=fam, param=v, alpha=a, graphs=len(g), Ns=g.N.nunique(), exponent=b,
                     ci=(hi - lo) / 2, cost_per_level=cost, t_100k=t100,
                     vs_flat_100k=t100 / flat.get(a, np.nan)))
tab = pd.DataFrame(rows)

lines += ['| family | value | α | graphs | N values | exponent | ± | cost per level | time at 100k | vs flat at 100k |',
          '|---|---|---|---|---|---|---|---|---|---|']
for r in tab.sort_values(['family', 'param', 'alpha']).itertuples():
    lines.append(f"| {r.family} | {'∞' if np.isinf(r.param) else f'{r.param:g}'} | {r.alpha:g} | {r.graphs} | "
                 f"{r.Ns} | {r.exponent:.3f} | {r.ci:.3f} | {r.cost_per_level:.2f} | {r.t_100k:.1f} | "
                 f"{r.vs_flat_100k:.2f} |")
text = '\n'.join(lines) + '\n'
with open(os.path.join(RES, '2026-11-05_internal_alpha10', 'summary_tables.md'), 'w', encoding='utf-8') as f:
    f.write(text)
print(text)
