"""
2026-11-04 follow-up on internal consensus under the alpha rule. Two checks:

  1. Where does the slowest disagreement mode sit? At q = 0.05 and alpha >= 3-4, internal consensus
     time sits at ~87 steps on every structure, alpha and N. The suspicion is a mode on the top
     levels, which alpha can't reach because top nodes have no superiors.
  2. Does the crossover continue past N = 10,000? On growing pyramids at alpha = 6 the hierarchy
     (q = 0.05) stays flat while the non-hierarchy (q = 1) slows with N; they meet at N = 10,000.

Structures are alpha_sweep/sweep_model.py's (rule B, 6 subordinates), with the growing depth
schedule continued at the same rate: 12 levels at N = 30,000 and 13 at 100,000. T comes from
results/2026-11-04_alpha_sweep_selfloop/T_table.csv; for N > 10,000 the N = 10,000 value is
reused (T for a given q moved by < 0.04 between 8 and 11 levels). Each graph's measured q is
recorded to check this.

For each graph and alpha: lambda_2 of W (the largest |eigenvalue| after the eigenvalue 1), its
right eigenvector v (the opinion pattern that persists longest), and how |v|^2 is spread over
the levels.

  python run_internal.py      -> results/2026-11-04_alpha_internal_slow_mode/slow_mode.csv
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
from scipy.sparse.linalg import eigs                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'alpha_sweep'))
import sweep_model as sm                                   # noqa: E402
from sweep_model import am, mdl, build_weight_matrix       # noqa: E402

OUT = os.path.join(sm.PROBE, 'results', '2026-11-04_alpha_internal_slow_mode')
T_TABLE = os.path.join(sm.PROBE, 'results', '2026-11-04_alpha_sweep_selfloop', 'T_table.csv')
BASE_SEED = 20261105
WORKERS = 4                                                # ~1 GB per worker at N = 100,000
STRUCTURES = ['fixed_r0.8', 'grow_r0.8', 'grow_r0.65']
EXTRA_DEPTH = {30_000: 12, 100_000: 13}                    # growing schedule, same rate
REPS = {300: 8, 1000: 8, 3000: 6, 10_000: 6, 30_000: 4, 100_000: 3}
Q_INDEX = {0: 0.05, 2: 1.0}
ALPHAS = [1.0, 2.0, 3.0, 4.0, 6.0]


def n_levels(st, N):
    r, grow = sm.STRUCTURES[st]
    if not grow:
        return 8
    return EXTRA_DEPTH[N] if N in EXTRA_DEPTH else sm.GROW_DEPTH[r][N]


def level_share(vec, levels, n_lv):
    v = np.abs(vec) ** 2
    v /= v.sum()
    return np.bincount(levels - 1, weights=v, minlength=n_lv), 1 / (v ** 2).sum()


def slow_mode(Gc, alpha, levels, n_lv):
    '''lambda_2 of the alpha-rule W (largest |eigenvalue| after 1), the next eigenvalue's modulus,
    |v_2|^2 per level and v_2's participation; then the same for the slowest non-oscillating mode
    (largest real part after 1), which differs from lambda_2 when lambda_2 is negative or complex.'''
    W = build_weight_matrix(Gc, alpha)
    n = W.shape[0]
    v0 = np.ones(n) + np.linspace(0, 1, n)
    vals, vecs = eigs(W, k=6, which='LM', ncv=min(n - 1, 60), tol=1e-10, maxiter=100_000, v0=v0)
    order = np.argsort(-np.abs(vals))
    vals, vecs = vals[order], vecs[:, order]
    assert abs(vals[0] - 1) < 1e-8, vals[0]
    share, part = level_share(vecs[:, 1], levels, n_lv)
    pv, pvecs = eigs(W, k=3, which='LR', ncv=min(n - 1, 60), tol=1e-10, maxiter=100_000, v0=v0)
    order = np.argsort(-pv.real)
    pv, pvecs = pv[order], pvecs[:, order]
    assert abs(pv[0] - 1) < 1e-8, pv[0]
    pshare, ppart = level_share(pvecs[:, 1], levels, n_lv)
    return vals[1], np.abs(vals[2]), share, part, pv[1], pshare, ppart


def run_one(task):
    st, N, qi, T, rep = task
    rng = np.random.default_rng([BASE_SEED, STRUCTURES.index(st), N, qi, rep])
    t0 = time.time()
    D = n_levels(st, N)
    P = mdl.prepare(am.rule_b_on_levels(mdl.level_sizes(N, D, sm.STRUCTURES[st][0]), sm.C, T, rng))
    Gc = P['G']
    am.tag_asymmetric(Gc)
    levels = np.array([Gc.nodes[n]['level'] for n in range(P['n'])])
    rows = []
    for alpha in ALPHAS:
        row = dict(structure=st, r=sm.STRUCTURES[st][0], depth=D, N=N, N_actual=P['n'],
                   n_dropped=P['n_dropped'], q_target=Q_INDEX[qi], T=T, q=P['q'], rep=rep,
                   alpha=alpha, error='')
        try:
            lam2, lam3, share, part, lp, pshare, ppart = slow_mode(Gc, alpha, levels, D)
            row.update(internal_time=mdl.LN_DELTA / np.log(abs(lam2)), lam2_re=lam2.real,
                       lam2_im=lam2.imag, lam2_abs=abs(lam2), lam3_abs=lam3,
                       top1_share=share[-1], top2_share=share[-2:].sum(), bottom_share=share[0],
                       participation=part, level_shares=';'.join(f'{x:.4g}' for x in share),
                       # slowest non-oscillating mode
                       nonosc_time=mdl.LN_DELTA / np.log(abs(lp)), lamP_re=lp.real, lamP_im=lp.imag,
                       nonosc_top2_share=pshare[-2:].sum(), nonosc_bottom_share=pshare[0],
                       nonosc_participation=ppart,
                       nonosc_level_shares=';'.join(f'{x:.4g}' for x in pshare))
        except Exception as e:                             # keep going; the row records the failure
            row['error'] = repr(e)[:200]
        rows.append(row)
    for row in rows:
        row['seconds'] = time.time() - t0
    return rows


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    table = pd.read_csv(T_TABLE)
    tasks = []
    for N in sorted(REPS, reverse=True):                   # big graphs first for load balance
        for st in STRUCTURES:
            for qi in Q_INDEX:
                N_T = min(N, 10_000)
                T = table.query('structure == @st and N == @N_T and q_index == @qi')['T'].iloc[0]
                tasks += [(st, N, qi, float(T), rep) for rep in range(REPS[N])]
    t0, rows = time.time(), []
    with get_context('spawn').Pool(WORKERS) as pool:
        for i, out in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1), 1):
            rows += out
            if i % 10 == 0 or i == len(tasks):
                print(f"  graphs {i}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'slow_mode.csv'), index=False)
    print(f"done: {len(df)} rows, {(df.error != '').sum()} errors, {time.time() - t0:.0f}s")
