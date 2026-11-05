"""
Driver for the 2026-10-29 overnight tests (see README.md in this folder).

Phase 1: q-calibration tables (median q on a T grid, per depth and size) and test 1.
Phase 2: tests 2, 3 and 4, using T values read off the calibration tables.

Every graph is one task with its own seed, so results don't depend on scheduling.
Rows are appended to one CSV per test as tasks finish, and finished tasks / tables
are skipped on restart, so the run can be resumed after a crash by rerunning it.

Usage:
    python run_all.py              # full run into results/2026-10-29_ruleB_pyramid_weights/
    python run_all.py --smoke DIR  # tiny version of everything, written to DIR
"""
import os
for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_var, '1')   # one BLAS thread per worker; parallelism comes from processes

import sys
import json
import time
import ctypes
import argparse
import threading
import traceback
from multiprocessing import get_context

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import model as mdl   # noqa: E402

N_RANGE = np.unique(np.round(np.logspace(2, 4, 20)).astype(int)).tolist()   # same sizes as simulation.py
T_GRID = np.unique(np.round(np.r_[np.arange(0.22, 0.40, 0.01), np.arange(0.40, 0.80, 0.025),
                                  np.arange(0.80, 1.60, 0.05), np.arange(1.6, 2.81, 0.1)], 4)).tolist()
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0]

FULL = dict(
    out_dir=os.path.join(PROBE, 'results', '2026-10-29_ruleB_pyramid_weights'),
    base_seed=20261029, workers=9, max_hours=6.0,
    D=8, r=0.8, c=6, gamma=0.01, frac=0.10,
    N_range=N_RANGE, T_grid=T_GRID, cal_M_scale=5000, cal_M_min=4,
    tests={
        # 1: pure hierarchy (T = 0.1 puts every edge exactly one level down, q = 0)
        1: dict(name='test1_hierarchy_q0', T=0.1, reps=20,
                conditions=[(0.5, 'sorted'), (0.0, 'sorted')], rules=['top', 'bottom', 'random']),
        # 2: random 10% seeding over the q range
        2: dict(name='test2_random_seeding', q=np.geomspace(0.05, 1.0, 10).tolist(), reps=10,
                conditions=[(0.5, 'sorted')], rules=['random', 'top', 'bottom']),
        # 3: influence distribution, sorted (5 sigmas) vs shuffled weights, same graphs
        3: dict(name='test3_weight_distribution', q=np.geomspace(0.05, 1.25, 8).tolist(), reps=10,
                conditions=[(s, 'sorted') for s in SIGMAS] + [(s, 'random') for s in SIGMAS if s > 0],
                rules=['random', 'top', 'bottom']),
        # 4: depth, r and c held fixed
        4: dict(name='test4_depth', depths=[4, 6, 8, 10, 12], N=[300, 1000, 3000, 10000],
                q=np.geomspace(0.05, 1.0, 10).tolist(), reps=5,
                conditions=[(0.5, 'sorted')], rules=['top', 'bottom', 'random']),
    },
)


def smoke_config(out_dir):
    cfg = json.loads(json.dumps(FULL))
    cfg['tests'] = {int(k): v for k, v in cfg['tests'].items()}
    cfg.update(out_dir=out_dir, workers=4, max_hours=0.5, N_range=[100, 300],
               T_grid=[0.22, 0.26, 0.30, 0.35, 0.45, 0.6, 0.8, 1.2, 1.8, 2.8], cal_M_scale=300)
    cfg['tests'][1].update(reps=2)
    cfg['tests'][2].update(q=[0.05, 0.5], reps=2)
    cfg['tests'][3].update(q=[0.05, 1.25], reps=2)
    cfg['tests'][4].update(depths=[4, 12], N=[300, 1000], q=[0.05, 1.0], reps=1)
    return cfg


# --- tasks (run in worker processes) ---------------------------------------

def calibration_task(task):
    '''q of M graphs at one grid T, for one (depth, table size).'''
    D, N, ti, T, M, cfg = task['D'], task['N'], task['ti'], task['T'], task['M'], task['cfg']
    rng = np.random.default_rng([cfg['base_seed'], 99, D, N, ti])
    return [mdl.prepare(mdl.create_rule_b(N, D, cfg['r'], cfg['c'], T, rng))['q'] for _ in range(M)]


def graph_task(task):
    '''One graph: every weighting and seeding rule of its test, all on the same graph.'''
    cfg, spec = task['cfg'], task['spec']
    D, N, qi, T, rep = task['D'], task['N'], task['qi'], task['T'], task['rep']
    rng = np.random.default_rng([cfg['base_seed'], task['test'], D, N, qi, rep])
    P = mdl.prepare(mdl.create_rule_b(N, D, cfg['r'], cfg['c'], T, rng))
    base = dict(test=task['test'], D=D, N=N, N_actual=P['n'], n_dropped=P['n_dropped'], q_index=qi,
                q_target=task['q_target'], T=T, q=P['q'], rep=rep)
    rows = []
    for sigma, placement in spec['conditions']:
        w = mdl.influence_weights(P['s'], sigma, placement, rng)
        d = P['K'] @ w
        t_int, lam2 = mdl.internal_consensus(P['K'], w, d)
        for rule in spec['rules']:
            S = mdl.pick_sources(rule, P['s'], cfg['frac'], rng)
            t_al, rate, approx, infl = mdl.alignment(P['K'], w, d, S, cfg['gamma'])
            rows.append(dict(base, sigma=sigma, placement=placement, internal_time=t_int,
                             internal_lam2=lam2, seed_rule=rule, n_sources=len(S), align_time=t_al,
                             align_rate=rate, approx_rate=approx, source_influence=infl))
    return rows


def worker(task):
    try:
        out = calibration_task(task) if task['kind'] == 'cal' else graph_task(task)
        return task, out, None
    except Exception:
        return task, None, traceback.format_exc()


# --- driver ------------------------------------------------------------------

class Run:
    def __init__(self, cfg):
        self.cfg = cfg
        self.out = cfg['out_dir']
        os.makedirs(os.path.join(self.out, 'calibration'), exist_ok=True)
        self.log_f = open(os.path.join(self.out, 'run_log.txt'), 'a', encoding='utf-8')
        self.failures = []

    def say(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + '\n')
        self.log_f.flush()

    # calibration tables
    def table_path(self, D, N):
        return os.path.join(self.out, 'calibration', f'D{D}_N{N}.npz')

    def needed_tables(self):
        cfg = self.cfg
        need = {(cfg['D'], min(N, 1000)) for N in cfg['N_range']}
        need |= {(D, min(N, 1000)) for D in cfg['tests'][4]['depths'] for N in cfg['tests'][4]['N']}
        return sorted(need)

    def T_for(self, D, N, q_targets):
        z = np.load(self.table_path(D, min(N, 1000)))
        return mdl.invert_q_table(q_targets, z['T_grid'], z['q_med'])

    # results files
    def csv_path(self, test):
        return os.path.join(self.out, self.cfg['tests'][test]['name'] + '.csv')

    def done_keys(self, test):
        path = self.csv_path(test)
        if not os.path.exists(path):
            return set()
        df = pd.read_csv(path, usecols=['D', 'N', 'q_index', 'rep'], on_bad_lines='skip')
        return set(map(tuple, df[['D', 'N', 'q_index', 'rep']].astype(int).values))

    def graph_tasks(self, test):
        cfg, spec = self.cfg, self.cfg['tests'][test]
        done = self.done_keys(test)
        tasks, skipped = [], []
        if test == 1:
            cells = [(cfg['D'], N, 0, 0.0, spec['T']) for N in cfg['N_range']]
        else:
            Ds = spec['depths'] if test == 4 else [cfg['D']]
            Ns = spec['N'] if test == 4 else cfg['N_range']
            cells = []
            for D in Ds:
                for N in Ns:
                    for qi, (qt, T) in enumerate(zip(spec['q'], self.T_for(D, N, spec['q']))):
                        if np.isnan(T):
                            skipped.append((D, N, round(qt, 4)))
                        else:
                            cells.append((D, N, qi, qt, float(T)))
        for D, N, qi, qt, T in cells:
            for rep in range(spec['reps']):
                if (D, N, qi, rep) not in done:
                    tasks.append(dict(kind='graph', test=test, spec=spec, cfg=self.cfg_light(),
                                      D=D, N=N, qi=qi, q_target=qt, T=T, rep=rep))
        if skipped:
            self.say(f"test {test}: q targets above the reachable range, skipped: {skipped}")
        if done:
            self.say(f"test {test}: {len(done)} graphs already done, resuming")
        return tasks

    def cfg_light(self):
        return {k: self.cfg[k] for k in ('base_seed', 'r', 'c', 'gamma', 'frac')}

    def run_pool(self, tasks, label):
        if not tasks:
            self.say(f"{label}: nothing to do")
            return
        tasks.sort(key=lambda t: -t['N'])                 # big graphs first for load balance
        pending_cal = {}
        for t in tasks:
            if t['kind'] == 'cal':
                pending_cal.setdefault((t['D'], t['N']), {})[t['ti']] = None
        t0, last, n_done = time.time(), 0.0, 0
        self.say(f"{label}: {len(tasks)} tasks on {self.cfg['workers']} workers")
        with get_context('spawn').Pool(self.cfg['workers']) as pool:
            for task, out, err in pool.imap_unordered(worker, tasks, chunksize=1):
                n_done += 1
                if err:
                    self.failures.append((task['kind'], {k: task[k] for k in ('D', 'N')}, err))
                    self.say(f"FAILED {task['kind']} D={task['D']} N={task['N']}: {err.strip().splitlines()[-1]}")
                elif task['kind'] == 'cal':
                    key = (task['D'], task['N'])
                    pending_cal[key][task['ti']] = out
                    if all(v is not None for v in pending_cal[key].values()):
                        self.save_table(key, pending_cal.pop(key))
                else:
                    path = self.csv_path(task['test'])
                    pd.DataFrame(out).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
                now = time.time()
                if now - last > 60 or n_done == len(tasks):
                    last = now
                    el = now - t0
                    self.say(f"{label}: {n_done}/{len(tasks)} tasks, {el / 60:.1f} min elapsed, "
                             f"~{el / n_done * (len(tasks) - n_done) / 60:.1f} min left (rough)")
        for key in pending_cal:
            self.say(f"calibration table D={key[0]} N={key[1]} incomplete (a grid point failed); not saved")

    def save_table(self, key, by_ti):
        T_grid = np.array(self.cfg['T_grid'])
        q_all = [by_ti[i] for i in range(len(T_grid))]
        q_med = np.array([np.median(v) for v in q_all])
        np.savez(self.table_path(*key), T_grid=T_grid, q_med=q_med,
                 q_all=np.array([np.pad(np.array(v, float), (0, max(map(len, q_all)) - len(v)),
                                        constant_values=np.nan) for v in q_all]))
        self.say(f"calibration D={key[0]} N={key[1]} saved: q from {q_med.min():.3f} to {q_med.max():.3f}")


def keep_awake():
    try:   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while this process runs
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', metavar='DIR')
    args = ap.parse_args()
    cfg = smoke_config(args.smoke) if args.smoke else FULL
    run = Run(cfg)

    def watchdog():
        run.say(f"WATCHDOG: exceeded {cfg['max_hours']} h, exiting so the run can be inspected and resumed")
        os._exit(3)
    timer = threading.Timer(cfg['max_hours'] * 3600, watchdog)
    timer.daemon = True
    timer.start()

    with open(os.path.join(run.out, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=1, default=str)
    run.say(f"start: out_dir={run.out}, keep-awake={'on' if keep_awake() else 'unavailable'}")

    # phase 1: calibration tables + test 1 (test 1 uses a fixed T)
    cal_tasks = []
    for D, N in run.needed_tables():
        if os.path.exists(run.table_path(D, N)):
            continue
        M = max(cfg['cal_M_min'], round(cfg['cal_M_scale'] / N))
        cal_tasks += [dict(kind='cal', D=D, N=N, ti=ti, T=T, M=M, cfg=run.cfg_light())
                      for ti, T in enumerate(cfg['T_grid'])]
    run.run_pool(cal_tasks + run.graph_tasks(1), 'phase 1 (calibration + test 1)')
    missing = [k for k in run.needed_tables() if not os.path.exists(run.table_path(*k))]
    if missing:
        run.say(f"STOP: calibration tables missing {missing}; rerun to retry")
        sys.exit(2)

    # phase 2: tests 2-4
    run.run_pool(run.graph_tasks(2) + run.graph_tasks(3) + run.graph_tasks(4), 'phase 2 (tests 2-4)')

    run.say(f"finished with {len(run.failures)} failed tasks" if run.failures else "finished, no failures")
    if run.failures:
        with open(os.path.join(run.out, 'failures.txt'), 'w', encoding='utf-8') as f:
            for kind, where, err in run.failures:
                f.write(f"{kind} {where}\n{err}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
