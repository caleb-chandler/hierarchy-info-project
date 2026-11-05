"""
One row per condition: size exponents at the extreme hierarchy (q = 0.05) and the extreme
non-hierarchy (q = 1), and whether they diverge: gap >= 0.05 (about a 19% difference in time
across N = 300-10,000) with its 95% CI excluding zero.
Usage: python divergence_table.py   -> prints markdown, writes results/.../divergence_table.csv
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', '2026-10-29_ablation')
MIN_GAP = 0.05
OUTCOMES = [('internal_time', 'internal'), ('align_bottom', 'bottom'),
            ('align_top', 'top'), ('align_random', 'random')]
LEVELS = {'tree': 'tree', 'tree_cap8': 'tree, cap 8', 'tree_cap8_b24': 'tree, cap 8, 24% basal',
          'pyramid_grow': 'growing pyramid', 'pyramid': 'pyramid'}
EDGES = {'gppm': 'GPPM', 'gppm_matched': 'GPPM thinned', 'c6': '6 subordinates'}
WEIGHTS = {'alpha2': 'α = 2', 'sorted': 'sorted', 'equal': 'equal'}

df = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in ['ablation.csv', 'ablation_split.csv']])


def slope(h, y):
    X = np.c_[np.ones(len(h)), np.log(h.N)]
    b, res, *_ = np.linalg.lstsq(X, np.log(h[y].values), rcond=None)
    return b[1], np.sqrt(res[0] / (len(h) - 2) * np.linalg.inv(X.T @ X)[1, 1])


rows = []
for (lv, ed, wt), g in df.groupby(['levels', 'edges', 'weighting']):
    r = dict(levels=lv, edges=ed, weighting=wt)
    for y, name in OUTCOMES:
        (bh, sh), (be, se) = slope(g[g.q_index == 0], y), slope(g[g.q_index == 2], y)
        gap, ci = bh - be, 1.96 * np.hypot(sh, se)
        r.update({f'{name}_hier': bh, f'{name}_egal': be,
                  f'{name}_diverges': abs(gap) >= MIN_GAP and abs(gap) > ci})
    rows.append(r)
T = pd.DataFrame(rows)
T['o'] = (T.levels.map({k: i for i, k in enumerate(LEVELS)}) * 100
          + T.edges.map({k: i for i, k in enumerate(EDGES)}) * 10
          + T.weighting.map({k: i for i, k in enumerate(WEIGHTS)}))
T = T.sort_values('o').drop(columns='o')
T.to_csv(os.path.join(OUT, 'divergence_table.csv'), index=False)

print('| levels | edges | weights | internal: hier. | egal. | diverges | bottom-seeded: hier. | egal. '
      '| diverges | top-seeded: hier. | egal. | diverges |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for _, r in T.iterrows():
    tag = (' **(old)**' if (r.levels, r.edges, r.weighting) == ('tree', 'gppm', 'alpha2') else
           ' **(new)**' if (r.levels, r.edges, r.weighting) == ('pyramid', 'c6', 'sorted') else '')
    cells = []
    for _, name in OUTCOMES[:3]:
        cells += [f"{r[f'{name}_hier']:+.2f}", f"{r[f'{name}_egal']:+.2f}",
                  '**yes**' if r[f'{name}_diverges'] else 'no']
    print(f"| {LEVELS[r.levels]} | {EDGES[r.edges]} | {WEIGHTS[r.weighting]}{tag} | " + ' | '.join(cells) + ' |')
print(f"\nrandom seeding diverges in {T.random_diverges.sum()} of {len(T)} conditions; "
      f"its exponents span {T[['random_hier', 'random_egal']].min().min():+.3f} to "
      f"{T[['random_hier', 'random_egal']].max().max():+.3f}")
