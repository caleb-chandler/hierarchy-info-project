"""
2026-11-04 run: top-, bottom- and random-seeded alignment time on the same graphs as run_scaling.py
(same seeds, families, sizes and alphas, so rows pair graph-for-graph with scaling.csv).

Sources are the top / bottom 10% of nodes by trophic level s, or a random 10%; each gives a
gamma = 0.01 share of its attention to a fixed-opinion environment node (engine.build_weight_matrix).
Continuous-time alignment time is ln(1e6) / (1 - rho), rho = Perron root of the real-node block.

Bottom-seeded influence becomes far smaller than an eigensolver can resolve (down to ~1e-12 at
N = 100,000 and alpha = 6), so each row records both:
  t_exact   ln(1e6) / (1 - rho), used when 1 - rho > 1e-9
  t_est     ln(1e6) / (gamma pi(S)), the weak-coupling estimate, with pi from lazy power iteration
            (check_alignment.py: within 0.1-1% of t_exact bottom-seeded, 1-7% top-seeded)
and t_align is t_exact where it's trustworthy, else t_est (column 'method').

  python run_alignment.py   -> results/2026-11-04_alignment_T_vs_rewiring/alignment.csv
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
from run_scaling import FAMILIES, REPS, ALPHAS, BASE_SEED  # noqa: E402  (same graphs)

OUT = os.path.join(sm.PROBE, 'results', '2026-11-04_alignment_T_vs_rewiring')
CSV = os.path.join(OUT, 'alignment.csv')
WORKERS = 8
GAMMA, SOURCE_FRAC, RULES = 0.01, 0.10, ['top', 'random', 'bottom']
EXACT_MIN_GAP = 1e-9


def run_one(task):
    fam, pi_idx, N, rep = task
    par = FAMILIES[fam][pi_idx]
    rng = np.random.default_rng([BASE_SEED, 0 if fam == 'T' else 1, pi_idx, N, rep])   # as run_scaling
    t0 = time.time()
    P = mdl.prepare(sm.make_graph(fam, par, N, rng))
    Gc = P['G']
    am.tag_asymmetric(Gc)
    sources = {r: mdl.pick_sources(r, P['s'], SOURCE_FRAC, rng) for r in RULES}
    base = dict(family=fam, param_index=pi_idx, param=par, N=N, depth=sm.depth(N), N_actual=P['n'],
                n_dropped=P['n_dropped'], q=P['q'], rep=rep)
    rows = []
    for alpha in ALPHAS:
        try:
            pi, iters, conv = sm.influence(Gc, alpha, list(sources.values()))
        except Exception as e:                             # keep going; the rows record the failure
            pi, iters, conv, err = None, 0, False, repr(e)[:200]
        for r, S in sources.items():
            row = dict(base, alpha=alpha, seed_rule=r, n_sources=len(S), iterations=iters,
                       converged=conv, error='' if pi is not None else err)
            try:
                infl = pi[S].sum()
                gap = sm.alignment_gap(Gc, alpha, S, GAMMA)
                t_est = sm.LN_1E6 / (GAMMA * infl)
                t_exact = sm.LN_1E6 / gap if gap > 0 else np.inf
                exact_ok = gap > EXACT_MIN_GAP
                row.update(influence=infl, gap_exact=gap, t_exact=t_exact, t_est=t_est,
                           t_align=t_exact if exact_ok else t_est, method='exact' if exact_ok else 'estimate')
            except Exception as e:
                row['error'] = row['error'] or repr(e)[:200]
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
    print(f"done: {len(df)} rows, {(df.error.fillna('') != '').sum()} errors, "
          f"{(~df.converged).sum()} unconverged influence rows, {time.time() - t0:.0f}s")
