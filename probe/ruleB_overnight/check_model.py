"""
Correctness checks for model.py, run before the overnight jobs.

1. The generator reproduces the q(T) values of the earlier rule-B prototype.
2. With equal weights (sigma = 0) the weight matrix, internal consensus time and
   alignment time match engine.py's build_weight_matrix / compute_spectral_gap
   at alpha = 1 (the existing code with alpha switched off).
3. With unequal weights, the symmetric shortcuts match dense eigen-solves of the
   explicit matrices, and pi = w*d/sum matches the stationary distribution.
4. Sorting and seeding behave as intended.
5. Timing of one N = 10,000 graph with test 3's full set of conditions.
"""
import time
import numpy as np
from scipy.stats import spearmanr

import model as mdl
from engine import build_weight_matrix, compute_spectral_gap

rng = np.random.default_rng(2026)
GAMMA = 0.01

print("--- 1. q(T) at N=1000, 8 levels, r=0.8, c=6 (prototype: 0, 0.096, 0.418, 1.149) ---")
for T in [0.1, 0.31, 0.5, 1.5]:
    qs = [mdl.prepare(mdl.create_rule_b(1000, 8, 0.8, 6, T, rng))['q'] for _ in range(3)]
    print(f"T={T}: q = {np.round(qs, 3).tolist()}")

print("\n--- 2. sigma = 0 vs engine.py at alpha = 1 (N=300) ---")
for T in [0.1, 0.5]:
    P = mdl.prepare(mdl.create_rule_b(300, 8, 0.8, 6, T, rng))
    G, K, s, n = P['G'], P['K'], P['s'], P['n']
    for u, v in G.edges():
        G[u][v]['is_asymmetric'] = not G.has_edge(v, u)
    w = mdl.influence_weights(s, 0.0, 'sorted', rng)
    W, d = mdl.weight_matrix(K, w)
    W_eng = build_weight_matrix(G, 1.0)
    t_int, _ = mdl.internal_consensus(K, w, d)
    t_int_eng = compute_spectral_gap(W_eng)['predicted_convergence_time']
    print(f"T={T}: max|W - W_engine| = {abs(W - W_eng).max():.1e}; internal {t_int:.4f} vs engine {t_int_eng:.4f}")
    for rule in ['top', 'bottom', 'random']:
        S = mdl.pick_sources(rule, s, 0.1, rng)
        t_al = mdl.alignment(K, w, d, S, GAMMA)[0]
        t_eng = compute_spectral_gap(build_weight_matrix(G, 1.0, sources=S, gamma=GAMMA))['predicted_convergence_time']
        print(f"     {rule:<6} alignment {t_al:.4f} vs engine {t_eng:.4f} (rel diff {abs(t_al / t_eng - 1):.1e})")

print("\n--- 3. unequal weights vs dense solves (N=200) ---")
for T, sigma, placement in [(0.1, 1.0, 'sorted'), (0.5, 1.0, 'random'), (1.5, 2.0, 'sorted')]:
    P = mdl.prepare(mdl.create_rule_b(200, 8, 0.8, 6, T, rng))
    K, s, n = P['K'], P['s'], P['n']
    w = mdl.influence_weights(s, sigma, placement, rng)
    W, d = mdl.weight_matrix(K, w)
    Wd = W.toarray()
    ev, evec = np.linalg.eig(Wd.T)
    pi_dense = np.abs(evec[:, np.argmax(ev.real)].real)
    pi_dense /= pi_dense.sum()
    pi = w * d / (w * d).sum()
    lam2_dense = np.sort(np.abs(np.linalg.eigvals(Wd)))[-2]
    t_int, lam2 = mdl.internal_consensus(K, w, d)
    print(f"T={T} sigma={sigma} {placement}: row sums 1 within {np.abs(Wd.sum(1) - 1).max():.1e}; "
          f"max|pi - pi_dense| = {np.abs(pi - pi_dense).max():.1e}; |lam2| {lam2:.8f} vs dense {lam2_dense:.8f}")
    for rule in ['top', 'bottom', 'random']:
        S = mdl.pick_sources(rule, s, 0.1, rng)
        f = np.ones(n)
        f[S] = 1 - GAMMA
        W_env = np.zeros((n + 1, n + 1))
        W_env[:n, :n] = f[:, None] * Wd
        W_env[S, n] = GAMMA
        W_env[n, n] = 1.0
        rho_dense = np.sort(np.abs(np.linalg.eigvals(W_env)))[-2]
        t_al, rate, approx, _ = mdl.alignment(K, w, d, S, GAMMA)
        print(f"     {rule:<6} rho {1 - rate:.10f} vs dense {rho_dense:.10f}; "
              f"time {t_al:.1f}; influence estimate / exact rate = {approx / rate:.3f}")

print("\n--- 4. sorting and seeding (N=1000, T=0.31) ---")
P = mdl.prepare(mdl.create_rule_b(1000, 8, 0.8, 6, 0.31, rng))
s = P['s']
w = mdl.influence_weights(s, 0.5, 'sorted', rng)
wr = mdl.influence_weights(s, 0.5, 'random', rng)
top, bot = mdl.pick_sources('top', s, 0.1, rng), mdl.pick_sources('bottom', s, 0.1, rng)
others_top = np.setdiff1d(np.arange(P['n']), top)
others_bot = np.setdiff1d(np.arange(P['n']), bot)
print(f"sorted: spearman(s, w) = {spearmanr(s, w).statistic:.3f}, median w = {np.median(w):.3f}; "
      f"random: spearman = {spearmanr(s, wr).statistic:+.3f}; same values: {np.allclose(np.sort(w), np.sort(wr))}")
print(f"top {len(top)} nodes: min s {s[top].min():.3f} >= max s of rest {s[others_top].max():.3f}; "
      f"bottom {len(bot)}: max s {s[bot].max():.3f} <= min s of rest {s[others_bot].min():.3f}")

print("\n--- 5. timing, N=10,000, T=0.5, test 3 conditions (9 weightings x 3 seedings) ---")
t0 = time.perf_counter()
G = mdl.create_rule_b(10_000, 8, 0.8, 6, 0.5, rng)
t1 = time.perf_counter()
P = mdl.prepare(G)
t2 = time.perf_counter()
conds = [(sg, 'sorted') for sg in [0, 0.5, 1, 1.5, 2]] + [(sg, 'random') for sg in [0.5, 1, 1.5, 2]]
for sg, pl in conds:
    w = mdl.influence_weights(P['s'], sg, pl, rng)
    W, d = mdl.weight_matrix(P['K'], w)
    mdl.internal_consensus(P['K'], w, d)
    for rule in ['random', 'top', 'bottom']:
        mdl.alignment(P['K'], w, d, mdl.pick_sources(rule, P['s'], 0.1, rng), GAMMA)
t3 = time.perf_counter()
print(f"generate {t1 - t0:.2f}s, LCC + trophic + K {t2 - t1:.2f}s, 9 weightings x (internal + 3 alignments) {t3 - t2:.2f}s")
