"""
Does alignment pick up a size effect at larger gamma? New structure (pyramid + c6) and the old
one (tree + gppm), all three weightings, top / random / bottom seeding, gamma 0.01-0.9.
Reuses T from results/2026-10-29_ablation/T_table.csv.
Usage: python gamma_check.py   -> results/2026-10-29_ablation/gamma_sweep.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import time                                                # noqa: E402
from multiprocessing import get_context                    # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

import ablation_model as am                                # noqa: E402
from ablation_model import mdl                             # noqa: E402

OUT = os.path.join(am.PROBE, 'results', '2026-10-29_ablation')
STRUCTURES = [('pyramid', 'c6'), ('tree', 'gppm')]
GAMMAS = [0.01, 0.1, 0.3, 0.6, 0.9]
N_RANGE, Q_INDEX = [300, 1000, 3000, 10_000], [0, 2]       # q = 0.05 and 1.0
REPS = {300: 8, 1000: 8, 3000: 6, 10_000: 4}
RULES = ['top', 'random', 'bottom']


def one(task):
    lv, ed, N, qi, T, rep = task
    rng = np.random.default_rng([20261031, len(lv), len(ed), N, qi, rep])
    P = mdl.prepare(am.make_graph(lv, ed, N, T, rng))
    am.tag_asymmetric(P['G'])
    sources = {r: mdl.pick_sources(r, P['s'], 0.10, rng) for r in RULES}
    rows = []
    for wt in ['equal', 'sorted', 'alpha2']:
        if wt != 'alpha2':
            w = np.ones(P['n']) if wt == 'equal' else mdl.influence_weights(P['s'], 0.5, 'sorted', rng)
            _, d = mdl.weight_matrix(P['K'], w)
        for g in GAMMAS:
            for r, S in sources.items():
                t = (am.alignment_alpha(P['G'], 2.0, S, g, rng) if wt == 'alpha2'
                     else mdl.alignment(P['K'], w, d, S, g)[0])
                rows.append(dict(levels=lv, edges=ed, N=N, q_index=qi, q=P['q'], rep=rep,
                                 weighting=wt, gamma=g, seed_rule=r, align_time=t))
    return rows


if __name__ == '__main__':
    table = pd.read_csv(os.path.join(OUT, 'T_table.csv'))
    tasks = []
    for N in sorted(N_RANGE, reverse=True):
        for lv, ed in STRUCTURES:
            for qi in Q_INDEX:
                T = table.query('levels == @lv and edges == @ed and N == @N and q_index == @qi')['T'].iloc[0]
                tasks += [(lv, ed, N, qi, float(T), rep) for rep in range(REPS[N])]
    t0, rows = time.time(), []
    with get_context('spawn').Pool(9) as pool:
        for out in pool.imap_unordered(one, tasks, chunksize=1):
            rows += out
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'gamma_sweep.csv'), index=False)
    print(f"{len(tasks)} graphs, {time.time() - t0:.0f}s\n")

    # exponent of alignment time vs N, per gamma; cells "q=0.05 / q=1"
    for (lv, ed), g1 in df.groupby(['levels', 'edges'], sort=False):
        print(f"=== {lv} + {ed}: exponent vs N (q = 0.05 / q = 1.0) ===")
        print(f"{'weights':<8}{'seeding':<8}" + "".join(f"{'gamma ' + str(g):>16}" for g in GAMMAS))
        for (wt, r), g2 in g1.groupby(['weighting', 'seed_rule']):
            cells = []
            for g in GAMMAS:
                e = [np.polyfit(np.log(h.N), np.log(h.align_time), 1)[0]
                     for _, h in g2[g2.gamma == g].groupby('q_index')]
                cells.append(f"{e[0]:+.3f} / {e[1]:+.3f}")
            print(f"{wt:<8}{r:<8}" + "".join(f"{c:>16}" for c in cells))
        med = g1[g1.N == 10_000].groupby(['weighting', 'seed_rule', 'gamma', 'q_index']).align_time.median()
        print("median time at N = 10,000 (q = 0.05 / q = 1.0):")
        for (wt, r), m in med.groupby(level=[0, 1]):
            print(f"{wt:<8}{r:<8}" + "".join(
                f"{m[(wt, r, g, 0)]:>8.0f}/{m[(wt, r, g, 2)]:<7.0f}" for g in GAMMAS))
        print()
