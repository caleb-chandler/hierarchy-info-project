"""Checks before the ablation run. Usage: python check_ablation.py"""
import time

import numpy as np

import ablation_model as am
from ablation_model import mdl
from engine import build_weight_matrix, compute_spectral_gap

rng = np.random.default_rng(1)

# 1. rule B on pyramid levels reproduces model.create_rule_b exactly
G1 = mdl.create_rule_b(1000, 8, 0.8, 6, 0.4, np.random.default_rng(5))
G2 = am.rule_b_on_levels(mdl.level_sizes(1000, 8, 0.8), 6, 0.4, np.random.default_rng(5))
print("rule_b_on_levels == create_rule_b:", set(G1.edges()) == set(G2.edges()))

# 2. what each structure looks like (N = 1000, T = 0.3)
print(f"\nc_old = {am.C_OLD:.3f} edges per node")
print(f"{'structure':<15} {'levels':>6} {'basal%':>6} {'top 3 level sizes':>18} {'edges/N':>7} "
      f"{'min/max out (non-basal)':>23} {'q':>6} {'dropped':>7}")
for lv, ed in am.STRUCTURES:
    G = am.make_graph(lv, ed, 1000, 0.3, rng)
    levels = np.array([G.nodes[n]['level'] for n in G.nodes])
    k = np.bincount(levels)[1:]
    out = np.array([G.out_degree(n) for n in G.nodes])[levels > 1]
    P = mdl.prepare(G)
    print(f"{lv + '+' + ed:<15} {len(k):6d} {100 * k[0] / k.sum():6.1f} {str(k[-3:].tolist()):>18} "
          f"{G.number_of_edges() / 1000:7.2f} {out.min():>11d} / {out.max():<9d} {P['q']:6.3f} "
          f"{P['n_dropped']:7d}")

# 3. alpha = 1 through the general solver equals equal weights through the symmetric one
print("\nalpha=1 (general eigs) vs equal weights (symmetric), N = 1000, T = 0.5:")
for lv, ed in am.STRUCTURES:
    P = mdl.prepare(am.make_graph(lv, ed, 1000, 0.5, rng))
    am.tag_asymmetric(P['G'])
    w = np.ones(P['n'])
    _, d = mdl.weight_matrix(P['K'], w)
    S = mdl.pick_sources('top', P['s'], 0.1, rng)
    t_sym = mdl.internal_consensus(P['K'], w, d)[0]
    t_gen = am.internal_alpha(P['G'], 1.0, rng)[0]
    a_sym = mdl.alignment(P['K'], w, d, S, 0.01)[0]
    a_gen = am.alignment_alpha(P['G'], 1.0, S, 0.01, rng)
    print(f"  {lv + '+' + ed:<15} internal {t_sym:9.3f} vs {t_gen:9.3f}   "
          f"alignment {a_sym:10.3f} vs {a_gen:10.3f}")

# 4. alpha = 2 internal time vs engine.compute_spectral_gap (dense fallback allowed at N = 300)
print("\nalpha=2 internal time vs engine.compute_spectral_gap, N = 300:")
for lv, ed in am.STRUCTURES:
    P = mdl.prepare(am.make_graph(lv, ed, 300, 0.5, rng))
    am.tag_asymmetric(P['G'])
    ref = compute_spectral_gap(build_weight_matrix(P['G'], 2.0))['predicted_convergence_time']
    print(f"  {lv + '+' + ed:<15} {am.internal_alpha(P['G'], 2.0, rng)[0]:9.3f} vs {ref:9.3f}")

# 5. timing at N = 10,000
print("\ntiming at N = 10,000 (s): generate, prepare, 4 symmetric solves, 4 alpha solves")
for lv, ed in am.STRUCTURES:
    t0 = time.time()
    G = am.make_graph(lv, ed, 10_000, 0.5, rng)
    t1 = time.time()
    P = mdl.prepare(G)
    am.tag_asymmetric(P['G'])
    t2 = time.time()
    w = np.ones(P['n'])
    _, d = mdl.weight_matrix(P['K'], w)
    mdl.internal_consensus(P['K'], w, d)
    for rule in ['top', 'random', 'bottom']:
        mdl.alignment(P['K'], w, d, mdl.pick_sources(rule, P['s'], 0.1, rng), 0.01)
    t3 = time.time()
    am.internal_alpha(P['G'], 2.0, rng)
    for rule in ['top', 'random', 'bottom']:
        am.alignment_alpha(P['G'], 2.0, mdl.pick_sources(rule, P['s'], 0.1, rng), 0.01, rng)
    t4 = time.time()
    print(f"  {lv + '+' + ed:<15} {t1 - t0:5.2f} {t2 - t1:5.2f} {t3 - t2:5.2f} {t4 - t3:5.2f}")
