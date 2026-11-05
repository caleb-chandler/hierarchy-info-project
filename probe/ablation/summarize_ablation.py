"""
Tables for the ablation. Usage: python summarize_ablation.py
Writes results/2026-10-29_ablation/summary_long.csv and prints the factorial tables.

Per combination x outcome:
  exp_q*     slope of log(time) on log(N), N = 300-10,000, at each q target
  t10k_q*    median time at N = 10,000
  q_ratio    t10k at q = 0.05 / t10k at q = 1.0   (> 1: the hierarchy is slower)
  noise      median over (N, q) cells of the graph-to-graph SD of log10(time)
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'results', '2026-10-29_ablation')
OUTCOMES = ['internal_time', 'align_top', 'align_random', 'align_bottom']
NEW = dict(levels='pyramid', edges='c6', weighting='sorted')
OLD = dict(levels='tree', edges='gppm', weighting='alpha2')

df = pd.read_csv(os.path.join(OUT, 'ablation.csv'))
assert (df.error.fillna('') == '').all()
recs = []
for (lv, ed, wt), g in df.groupby(['levels', 'edges', 'weighting']):
    for y in OUTCOMES:
        r = dict(levels=lv, edges=ed, weighting=wt, outcome=y)
        for qt, h in g.groupby('q_target'):
            r[f'exp_q{qt:g}'] = np.polyfit(np.log(h.N), np.log(h[y]), 1)[0]
            r[f't10k_q{qt:g}'] = h.loc[h.N == 10_000, y].median()
        r['q_ratio'] = r['t10k_q0.05'] / r['t10k_q1']
        r['noise'] = g.groupby(['N', 'q_target'])[y].apply(lambda v: np.log10(v).std()).median()
        recs.append(r)
S = pd.DataFrame(recs)
S.to_csv(os.path.join(OUT, 'summary_long.csv'), index=False)


def label(r):
    old = [k for k in NEW if r[k] == OLD[k]]
    new = [k for k in NEW if r[k] == NEW[k]]
    if r['edges'] == 'gppm_matched' or r['weighting'] == 'equal':
        return 'extra'
    if len(old) == 0:
        return 'NEW (overnight)'
    if len(new) == 0:
        return 'OLD'
    if len(old) == 1:
        return f'new, {old[0]} undone'
    return f'old, only {new[0]} changed'


S['combo'] = S.apply(label, axis=1)
order = ['NEW (overnight)', 'new, levels undone', 'new, edges undone', 'new, weighting undone',
         'old, only levels changed', 'old, only edges changed', 'old, only weighting changed', 'OLD', 'extra']
S['o'] = S.combo.map({c: i for i, c in enumerate(order)})
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 200)
cols = ['combo', 'levels', 'edges', 'weighting', 'exp_q0.05', 'exp_q0.3', 'exp_q1',
        't10k_q0.05', 't10k_q1', 'q_ratio', 'noise']
for y in OUTCOMES:
    print(f"\n=== {y} ===")
    print(S[S.outcome == y].sort_values(['o', 'levels', 'edges', 'weighting'])[cols]
          .to_string(index=False, float_format=lambda v: f'{v:.3g}'))

d = df.groupby(['levels', 'edges', 'N']).agg(depth_med=('depth', 'median'), depth_min=('depth', 'min'),
                                            depth_max=('depth', 'max'), epn=('edges_per_node', 'mean'))
print("\n=== depth and edges per node ===")
print(d.round(2).to_string())
