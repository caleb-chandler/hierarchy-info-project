"""
The levels change split into depth, occupancy shape and basal share. Combines ablation.csv
(tree, pyramid) with ablation_split.csv. Usage: python summarize_split.py
Writes results/2026-10-29_ablation/summary_split.csv.
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', '2026-10-29_ablation')
LEVELS = ['tree', 'tree_cap8', 'tree_cap8_b24', 'pyramid_grow', 'pyramid']
df = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in ['ablation.csv', 'ablation_split.csv']])
df = df[df.levels.isin(LEVELS) & df.edges.isin(['gppm', 'c6'])]
assert (df.error.fillna('') == '').all()


def exps(g, y):
    e = [np.polyfit(np.log(h.N), np.log(h[y]), 1)[0] for _, h in g.groupby('q_target')]
    return f"{min(e):+.2f} to {max(e):+.2f}"


recs = []
for (lv, ed, wt), g in df.groupby(['levels', 'edges', 'weighting']):
    big = g[g.N == 10_000]
    r = dict(levels=lv, edges=ed, weighting=wt,
             depth_300=g[g.N == 300].depth.median(), depth_10k=big.depth.median(),
             internal_exp=exps(g, 'internal_time'),
             internal_q=big[big.q_index == 0].internal_time.median() / big[big.q_index == 2].internal_time.median(),
             internal_noise=g.groupby(['N', 'q_index']).internal_time.apply(lambda v: np.log10(v).std()).median())
    for s in ['top', 'bottom']:
        y = f'align_{s}'
        r[f'{s}_exp'] = exps(g, y)
        r[f'{s}_q'] = big[big.q_index == 0][y].median() / big[big.q_index == 2][y].median()
    recs.append(r)
S = pd.DataFrame(recs)
S['o'] = S.levels.map({lv: i for i, lv in enumerate(LEVELS)})
S = S.sort_values(['edges', 'weighting', 'o']).drop(columns='o')
S.to_csv(os.path.join(OUT, 'summary_split.csv'), index=False)
pd.set_option('display.width', 250)
for (ed, wt), g in S.groupby(['edges', 'weighting'], sort=False):
    print(f"\n=== edges {ed}, weights {wt} ===")
    print(g.drop(columns=['edges', 'weighting']).to_string(index=False, float_format=lambda v: f'{v:.3g}'))
