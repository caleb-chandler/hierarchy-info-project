"""Checks for the levels split. Usage: python check_split.py"""
import os

import numpy as np
import pandas as pd

import ablation_model as am
import run_ablation as ra

# 1. tree_attach + gppm_on_levels (no cap) reproduces generator.create_new edge for edge
for N, T in [(300, 0.3), (2000, 1.0)]:
    G1 = am.gen.create_new(N, am.B_FRAC, am.C_OLD, T, np.random.default_rng(11))[0]
    rng = np.random.default_rng(11)
    lv, te = am.tree_attach(N, am.B_FRAC, rng)
    G2 = am.gppm_on_levels(lv, round(am.C_OLD * N), T, rng, te)
    same_levels = all(G1.nodes[n]['level'] == G2.nodes[n]['level'] for n in range(N))
    print(f"N={N}: tree path == create_new: edges {set(G1.edges()) == set(G2.edges())}, "
          f"levels {same_levels}")

# 2. the refactor leaves earlier ablation rows unchanged
old = pd.read_csv(os.path.join(ra.OUT, 'ablation.csv'))
table = pd.read_csv(os.path.join(ra.OUT, 'T_table.csv'))
for lv, ed in [('pyramid', 'gppm'), ('tree', 'c6'), ('pyramid', 'gppm_matched')]:
    T = table.query('levels == @lv and edges == @ed and N == 1000 and q_index == 1')['T'].iloc[0]
    rows = ra.run_one((ra.ALL_STRUCTURES.index((lv, ed)), 1000, 1, float(T), 3))
    ref = old.query('levels == @lv and edges == @ed and N == 1000 and q_index == 1 and rep == 3')
    new = pd.DataFrame(rows).set_index('weighting').internal_time
    print(f"{lv}+{ed}: rerun matches ablation.csv:",
          np.allclose(new[ref.weighting].values, ref.internal_time.values, rtol=1e-8))

# 3. what the level types look like (median over 20 draws)
rng = np.random.default_rng(2)
print(f"\n{'levels':<15} {'N':>6} {'depth':>6} {'basal%':>7} {'top 3 level sizes':>22}")
for lv in ['tree', 'tree_cap8', 'tree_cap8_b24', 'pyramid_grow', 'pyramid']:
    for N in [300, 10_000]:
        ks = []
        for _ in range(20 if lv.startswith('tree') else 1):
            if lv.startswith('tree'):
                b = 0.24 if lv.endswith('_b24') else am.B_FRAC
                ks.append(am.tree_levels(N, b, rng, cap=8 if '_cap8' in lv else None))
            else:
                ks.append(am.mdl.level_sizes(N, am.GROW_DEPTH[N] if lv == 'pyramid_grow' else 8, 0.8))
        depth = int(np.median([len(k) for k in ks]))
        k = next(k for k in ks if len(k) == depth)
        print(f"{lv:<15} {N:6d} {depth:6d} {100 * np.median([k[0] / k.sum() for k in ks]):7.1f} "
              f"{str(k[-3:].tolist()):>22}")
