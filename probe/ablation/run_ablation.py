"""
Ablation run: every combination of old/new levels, edges and weights.

  python run_ablation.py calibrate [split]   T tables per structure and size (median q of several graphs)
  python run_ablation.py run [split]         internal consensus and alignment for every graph x weighting

Without 'split' it runs the main factorial (+ the density extra); with 'split' it runs the
structures that split the levels change, into files with a _split suffix.

Every graph is evaluated under all three weightings (equal, sorted sigma = 0.5, alpha = 2) with
the same sources, so the weight comparison is within-graph.
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

import ablation_model as am                                # noqa: E402
from ablation_model import mdl                             # noqa: E402

OUT = os.path.join(am.PROBE, 'results', '2026-10-29_ablation')
BASE_SEED = 20261030
WORKERS = 9
ALL_STRUCTURES = am.STRUCTURES + am.EXTRA_STRUCTURES + am.SPLIT_STRUCTURES   # index = seed component
SETS = {'main': (am.STRUCTURES + am.EXTRA_STRUCTURES, ''), 'split': (am.SPLIT_STRUCTURES, '_split')}
STRUCTURES, SUFFIX = SETS['main']
N_RANGE = [300, 1000, 3000, 10_000]
Q_TARGETS = [0.05, 0.3, 1.0]
T_GRID = np.geomspace(0.1, 4.0, 18)
CAL_M = {300: 10, 1000: 6, 3000: 3, 10_000: 2}
REPS = {300: 16, 1000: 16, 3000: 12, 10_000: 10}
GAMMA, SOURCE_FRAC, RULES = 0.01, 0.10, ['top', 'random', 'bottom']


def calibrate_one(task):
    si, ti, N, m = task
    lv, ed = ALL_STRUCTURES[si]
    rng = np.random.default_rng([BASE_SEED, 99, si, N, ti, m])
    P = mdl.prepare(am.make_graph(lv, ed, N, T_GRID[ti], rng))
    return dict(levels=lv, edges=ed, N=N, T=T_GRID[ti], m=m, q=P['q'])


def run_one(task):
    si, N, qi, T, rep = task
    lv, ed = ALL_STRUCTURES[si]
    rng = np.random.default_rng([BASE_SEED, si, N, qi, rep])
    t0 = time.time()
    P = mdl.prepare(am.make_graph(lv, ed, N, T, rng))
    Gc = P['G']
    am.tag_asymmetric(Gc)
    levels = np.array([Gc.nodes[n]['level'] for n in range(P['n'])])
    sources = {r: mdl.pick_sources(r, P['s'], SOURCE_FRAC, rng) for r in RULES}
    base = dict(levels=lv, edges=ed, N=N, N_actual=P['n'], n_dropped=P['n_dropped'], q_index=qi,
                q_target=Q_TARGETS[qi], T=T, q=P['q'], depth=int(levels.max()),
                edges_per_node=Gc.number_of_edges() / P['n'], rep=rep)
    rows = []
    for weighting in ['equal', 'sorted', 'alpha2']:
        row = dict(base, weighting=weighting, error='')
        try:
            if weighting == 'alpha2':
                row['internal_time'] = am.internal_alpha(Gc, am.ALPHA_OLD, rng)[0]
                for r, S in sources.items():
                    row[f'align_{r}'] = am.alignment_alpha(Gc, am.ALPHA_OLD, S, GAMMA, rng)
            else:
                w = (np.ones(P['n']) if weighting == 'equal'
                     else mdl.influence_weights(P['s'], am.SIGMA_NEW, 'sorted', rng))
                _, d = mdl.weight_matrix(P['K'], w)
                row['internal_time'] = mdl.internal_consensus(P['K'], w, d)[0]
                for r, S in sources.items():
                    row[f'align_{r}'] = mdl.alignment(P['K'], w, d, S, GAMMA)[0]
        except Exception as e:                             # keep going; the row records the failure
            row['error'] = repr(e)[:200]
        rows.append(row)
    for row in rows:
        row['seconds'] = time.time() - t0
    return rows


def pool_map(fn, tasks):
    with get_context('spawn').Pool(WORKERS) as pool:
        for i, out in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
            yield i, out


def calibrate():
    tasks = [(ALL_STRUCTURES.index(st), ti, N, m) for N in sorted(N_RANGE, reverse=True)
             for st in STRUCTURES for ti in range(len(T_GRID)) for m in range(CAL_M[N])]
    t0, recs = time.time(), []
    for i, rec in pool_map(calibrate_one, tasks):
        recs.append(rec)
        if i % 200 == 0:
            print(f"  calibration {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    cal = pd.DataFrame(recs).sort_values(['levels', 'edges', 'N', 'T', 'm'])
    cal.to_csv(os.path.join(OUT, f'calibration{SUFFIX}.csv'), index=False)

    table = []
    for (lv, ed, N), g in cal.groupby(['levels', 'edges', 'N']):
        med = g.groupby('T').q.median()
        Ts = mdl.invert_q_table(Q_TARGETS, med.index.values, med.values)
        for qi, (qt, T) in enumerate(zip(Q_TARGETS, Ts)):
            table.append(dict(levels=lv, edges=ed, N=N, q_index=qi, q_target=qt, T=T,
                              q_min=med.min(), q_max=med.max()))
    table = pd.DataFrame(table)
    table.to_csv(os.path.join(OUT, f'T_table{SUFFIX}.csv'), index=False)
    print(table.pivot_table(index=['levels', 'edges', 'N'], columns='q_target', values='T').round(3))
    print(table.groupby(['levels', 'edges', 'N'])[['q_min', 'q_max']].first().round(3))


def run():
    table = pd.read_csv(os.path.join(OUT, f'T_table{SUFFIX}.csv'))
    tasks = []
    for N in sorted(N_RANGE, reverse=True):
        for lv, ed in STRUCTURES:
            si = ALL_STRUCTURES.index((lv, ed))
            for qi in range(len(Q_TARGETS)):
                T = table.query('levels == @lv and edges == @ed and N == @N and q_index == @qi')['T'].iloc[0]
                if np.isnan(T):
                    continue
                tasks += [(si, N, qi, float(T), rep) for rep in range(REPS[N])]
    t0, rows = time.time(), []
    for i, out in pool_map(run_one, tasks):
        rows += out
        if i % 50 == 0 or i == len(tasks):
            print(f"  graphs {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f'ablation{SUFFIX}.csv'), index=False)
    print(f"done: {len(df)} rows, {(df.error != '').sum()} errors, {time.time() - t0:.0f}s")


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    STRUCTURES, SUFFIX = SETS[sys.argv[2] if len(sys.argv) > 2 else 'main']
    {'calibrate': calibrate, 'run': run}[sys.argv[1]]()
