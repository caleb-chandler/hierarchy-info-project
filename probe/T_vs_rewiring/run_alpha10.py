"""
2026-11-05 quick run: continuous-time internal consensus at alpha = 10 on a subset of the 11-04 graphs
(same seeds as run_scaling.py, so the same graphs). Hierarchy and flat first, then the disorder peak.
Rows are appended as graphs finish; rerunning skips finished graphs.

  python run_alpha10.py      -> results/2026-11-05_internal_alpha10/scaling_alpha10.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import time                                                # noqa: E402
from multiprocessing import get_context                    # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

import scaling_model as sm                                 # noqa: E402
from scaling_model import am, mdl                          # noqa: E402
from run_scaling import FAMILIES, REPS, BASE_SEED, WORKERS  # noqa: E402

OUT = os.path.join(sm.PROBE, 'results', '2026-11-05_internal_alpha10')
CSV = os.path.join(OUT, 'scaling_alpha10.csv')
ALPHA = 10.0
PRIORITY = [[('T', 0.1), ('T', np.inf)],                   # hierarchy, flat
            [('T', 1.5), ('T', 1.0), ('T', 2.0), ('p', 0.03), ('p', 0.1)]]


def run_one(task):
    fam, pi, N, rep = task
    par = FAMILIES[fam][pi]
    rng = np.random.default_rng([BASE_SEED, 0 if fam == 'T' else 1, pi, N, rep])
    t0 = time.time()
    P = mdl.prepare(sm.make_graph(fam, par, N, rng))
    Gc = P['G']
    am.tag_asymmetric(Gc)
    row = dict(family=fam, param_index=pi, param=par, N=N, depth=sm.depth(N), N_actual=P['n'],
               n_dropped=P['n_dropped'], q=P['q'], rep=rep, alpha=ALPHA, error='')
    try:
        t, lam = sm.continuous_time(Gc, ALPHA)
        row.update(t_cont=t, lamP_re=lam.real, lamP_im=lam.imag)
    except Exception as e:
        row['error'] = repr(e)[:200]
    row['seconds'] = time.time() - t0
    return [row]


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(CSV):
        d = pd.read_csv(CSV, usecols=['family', 'param_index', 'N', 'rep'])
        done = set(map(tuple, d.drop_duplicates().values.tolist()))
    tasks = [(fam, FAMILIES[fam].index(par), N, rep) for group in PRIORITY
             for N in sorted(REPS, reverse=True) for fam, par in group
             for rep in range(REPS[N])]
    tasks = [t for t in tasks if t not in done]
    print(f"{len(tasks)} graphs to run ({len(done)} already done)", flush=True)
    t0 = time.time()
    with get_context('spawn').Pool(WORKERS) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1), 1):
            pd.DataFrame(rows).to_csv(CSV, mode='a', header=not os.path.exists(CSV), index=False)
            if i % 10 == 0 or i == len(tasks):
                print(f"  graphs {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done: {time.time() - t0:.0f}s")
