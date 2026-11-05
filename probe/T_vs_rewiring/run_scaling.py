"""
2026-11-04 run: continuous-time internal consensus vs N on the top-size pyramid (r = 0.65, top 15),
with disorder from the kernel width T or from rewiring fraction p. See scaling_model.py.

Every graph is evaluated under every alpha. Rows are appended to scaling.csv as graphs finish, so
a crash loses nothing done; rerunning skips finished graphs.

  python run_scaling.py      -> results/2026-11-04_internal_T_vs_rewiring/scaling.csv
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')              # one BLAS thread per worker
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import time                                                # noqa: E402
from multiprocessing import get_context                    # noqa: E402

import numpy as np                                         # noqa: E402
import pandas as pd                                        # noqa: E402

import scaling_model as sm                                 # noqa: E402
from scaling_model import am, mdl                          # noqa: E402

OUT = os.path.join(sm.PROBE, 'results', '2026-11-04_internal_T_vs_rewiring')
CSV = os.path.join(OUT, 'scaling.csv')
BASE_SEED = 20261106
WORKERS = 8                                                # ~1 GB per worker at N = 100,000
FAMILIES = {'T': [0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, np.inf],
            'p': [0.0, 0.003, 0.01, 0.03, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]}
REPS = {300: 8, 1000: 8, 3000: 6, 10_000: 4, 30_000: 3, 100_000: 2}
ALPHAS = [1.0, 2.0, 3.0, 4.0, 6.0]


def run_one(task):
    fam, pi, N, rep = task
    par = FAMILIES[fam][pi]
    rng = np.random.default_rng([BASE_SEED, 0 if fam == 'T' else 1, pi, N, rep])
    t0 = time.time()
    P = mdl.prepare(sm.make_graph(fam, par, N, rng))
    Gc = P['G']
    am.tag_asymmetric(Gc)
    base = dict(family=fam, param_index=pi, param=par, N=N, depth=sm.depth(N), N_actual=P['n'],
                n_dropped=P['n_dropped'], q=P['q'], rep=rep, **sm.tie_stats(Gc))
    rows = []
    for alpha in ALPHAS:
        row = dict(base, alpha=alpha, error='')
        try:
            t, lam = sm.continuous_time(Gc, alpha)
            row.update(t_cont=t, lamP_re=lam.real, lamP_im=lam.imag)
        except Exception as e:                             # keep going; the row records the failure
            row['error'] = repr(e)[:200]
        rows.append(row)
    for row in rows:
        row['seconds'] = time.time() - t0
    return rows


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(CSV):
        d = pd.read_csv(CSV, usecols=['family', 'param_index', 'N', 'rep'])
        done = set(map(tuple, d.drop_duplicates().values.tolist()))
    tasks = [(fam, pi, N, rep) for N in sorted(REPS, reverse=True)   # big graphs first
             for fam, pars in FAMILIES.items() for pi in range(len(pars))
             for rep in range(REPS[N]) if (fam, pi, N, rep) not in done]
    print(f"{len(tasks)} graphs to run ({len(done)} already done)", flush=True)
    t0 = time.time()
    with get_context('spawn').Pool(WORKERS) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1), 1):
            pd.DataFrame(rows).to_csv(CSV, mode='a', header=not os.path.exists(CSV), index=False)
            if i % 25 == 0 or i == len(tasks):
                print(f"  graphs {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.read_csv(CSV)
    print(f"done: {len(df)} rows, {(df.error.fillna('') != '').sum()} errors, {time.time() - t0:.0f}s")
