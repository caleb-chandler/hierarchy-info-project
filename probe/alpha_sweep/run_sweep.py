"""
2026-11-04 run: alpha sweep, alpha * r threshold, and lognormal weights with the self-tie at 1.

  python run_sweep.py calibrate   T tables for the two new structures (grow_r0.65, grow_r0.5);
                                  the r = 0.8 structures reuse the ablation's T tables
  python run_sweep.py run         every graph under every alpha (all structures) and every
                                  lognormal weighting x self-tie rule (the two r = 0.8 structures)

Every graph is evaluated under every weighting with the same sources, so all weight comparisons
are within-graph. Outputs go to results/2026-11-04_alpha_sweep_selfloop/.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')              # one BLAS thread per worker
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys                                                 # noqa: E402
import time                                                # noqa: E402
from multiprocessing import get_context                    # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

import sweep_model as sm                                   # noqa: E402
from sweep_model import am, mdl                            # noqa: E402

OUT = os.path.join(sm.PROBE, 'results', '2026-11-04_alpha_sweep_selfloop')
ABL = os.path.join(sm.PROBE, 'results', '2026-10-29_ablation')
BASE_SEED = 20261104
WORKERS = 9
STRUCTURES = list(sm.STRUCTURES)                           # index = seed component
NEW = ['grow_r0.65', 'grow_r0.5']                          # need calibrating
REUSED = {'fixed_r0.8': ('T_table.csv', 'pyramid'), 'grow_r0.8': ('T_table_split.csv', 'pyramid_grow')}
LOGNORMAL = ['fixed_r0.8', 'grow_r0.8']
N_RANGE = [300, 1000, 3000, 10_000]
Q_TARGETS = [0.05, 0.3, 1.0]
T_GRID = np.geomspace(0.1, 4.0, 18)                        # the ablation's grid
CAL_M = {300: 10, 1000: 6, 3000: 3, 10_000: 2}
REPS = {300: 16, 1000: 16, 3000: 12, 10_000: 10}
GAMMA, SOURCE_FRAC, RULES = 0.01, 0.10, ['top', 'random', 'bottom']


def calibrate_one(task):
    si, ti, N, m = task
    rng = np.random.default_rng([BASE_SEED, 99, si, N, ti, m])
    P = mdl.prepare(sm.make_graph(STRUCTURES[si], N, T_GRID[ti], rng))
    return dict(structure=STRUCTURES[si], N=N, T=T_GRID[ti], m=m, q=P['q'])


def run_one(task):
    si, N, qi, T, rep = task
    st = STRUCTURES[si]
    rng = np.random.default_rng([BASE_SEED, si, N, qi, rep])
    t0 = time.time()
    P = mdl.prepare(sm.make_graph(st, N, T, rng))
    Gc = P['G']
    am.tag_asymmetric(Gc)
    levels = np.array([Gc.nodes[n]['level'] for n in range(P['n'])])
    sources = {r: mdl.pick_sources(r, P['s'], SOURCE_FRAC, rng) for r in RULES}
    base = dict(structure=st, r=sm.STRUCTURES[st][0], depth=int(levels.max()), N=N, N_actual=P['n'],
                n_dropped=P['n_dropped'], q_index=qi, q_target=Q_TARGETS[qi], T=T, q=P['q'], rep=rep)

    alpha_rows = []
    for alpha in sm.ALPHAS:
        row = dict(base, alpha=alpha, alpha_r=alpha * base['r'], error='')
        try:
            row['internal_time'] = am.internal_alpha(Gc, alpha, rng)[0]
            pi = sm.influence_alpha(Gc, alpha)
            for r, S in sources.items():
                row[f'align_{r}'] = am.alignment_alpha(Gc, alpha, S, GAMMA, rng)
                row[f'infl_{r}'] = pi[S].sum()
            row['level_infl'] = ';'.join(f'{x:.5g}' for x in np.bincount(levels - 1, weights=pi))
        except Exception as e:                             # keep going; the row records the failure
            row['error'] = repr(e)[:200]
        alpha_rows.append(row)

    logn_rows = []
    if st in LOGNORMAL:
        for sigma in sm.SIGMAS:
            for placement in (['sorted'] if sigma == 0 else ['sorted', 'random']):
                w = mdl.influence_weights(P['s'], sigma, placement, rng)
                for self_rule in ['own', 'one']:          # same w, sources and graph for both
                    row = dict(base, sigma=sigma, placement=placement, self_rule=self_rule, error='')
                    try:
                        K, d = sm.tie_matrix(P['K'], w, self_rule)
                        row['internal_time'] = mdl.internal_consensus(K, w, d)[0]
                        row['max_self_share'] = ((w if self_rule == 'own' else 1) / d).max()
                        for r, S in sources.items():
                            t, _, _, infl = mdl.alignment(K, w, d, S, GAMMA)
                            row[f'align_{r}'], row[f'infl_{r}'] = t, infl
                    except Exception as e:
                        row['error'] = repr(e)[:200]
                    logn_rows.append(row)

    for row in alpha_rows + logn_rows:
        row['seconds'] = time.time() - t0
    return alpha_rows, logn_rows


def pool_map(fn, tasks):
    with get_context('spawn').Pool(WORKERS) as pool:
        for i, out in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
            yield i, out


def calibrate():
    tasks = [(STRUCTURES.index(st), ti, N, m) for N in sorted(N_RANGE, reverse=True)
             for st in NEW for ti in range(len(T_GRID)) for m in range(CAL_M[N])]
    t0, recs = time.time(), []
    for i, rec in pool_map(calibrate_one, tasks):
        recs.append(rec)
        if i % 100 == 0:
            print(f"  calibration {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    cal = pd.DataFrame(recs).sort_values(['structure', 'N', 'T', 'm'])
    cal.to_csv(os.path.join(OUT, 'calibration.csv'), index=False)

    table = []
    for (st, N), g in cal.groupby(['structure', 'N']):
        med = g.groupby('T').q.median()
        Ts = mdl.invert_q_table(Q_TARGETS, med.index.values, med.values)
        for qi, (qt, T) in enumerate(zip(Q_TARGETS, Ts)):
            table.append(dict(structure=st, N=N, q_index=qi, q_target=qt, T=T,
                              q_min=med.min(), q_max=med.max(), source='calibrated here'))
    for st, (fname, lv) in REUSED.items():
        old = pd.read_csv(os.path.join(ABL, fname)).query('levels == @lv and edges == "c6"')
        for _, o in old.iterrows():
            table.append(dict(structure=st, N=o.N, q_index=o.q_index, q_target=o.q_target, T=o['T'],
                              q_min=o.q_min, q_max=o.q_max, source=f'2026-10-29_ablation/{fname}'))
    table = pd.DataFrame(table)
    table.to_csv(os.path.join(OUT, 'T_table.csv'), index=False)
    print(table.pivot_table(index=['structure', 'N'], columns='q_target', values='T').round(3))
    print(table.groupby(['structure', 'N'])[['q_min', 'q_max']].first().round(3))


def run():
    table = pd.read_csv(os.path.join(OUT, 'T_table.csv'))
    tasks = []
    for N in sorted(N_RANGE, reverse=True):
        for st in STRUCTURES:
            for qi in range(len(Q_TARGETS)):
                T = table.query('structure == @st and N == @N and q_index == @qi')['T'].iloc[0]
                if np.isnan(T):
                    continue
                tasks += [(STRUCTURES.index(st), N, qi, float(T), rep) for rep in range(REPS[N])]
    t0, alpha_rows, logn_rows = time.time(), [], []
    for i, (a, l) in pool_map(run_one, tasks):
        alpha_rows += a
        logn_rows += l
        if i % 50 == 0 or i == len(tasks):
            print(f"  graphs {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    for name, rows in [('alpha.csv', alpha_rows), ('lognormal.csv', logn_rows)]:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, name), index=False)
        print(f"{name}: {len(df)} rows, {(df.error != '').sum()} errors")
    print(f"done, {time.time() - t0:.0f}s")


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    {'calibrate': calibrate, 'run': run}[sys.argv[1]]()
