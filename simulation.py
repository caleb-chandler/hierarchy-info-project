import os
import numpy as np
from generator import generator, valid_tree_sizes
from engine import run_trial
import pickle

# --- parameters ---
C = 6
b = int(C - 1)  # branching factor
N_MIN, N_MAX = 10, 10_000
N_POINTS = 20  # approximate number of sizes to test
weight_dist = None
n_trials = 20

structures = ['control', 'alternative', 'hierarchy']

# --- compute N_range from valid tree sizes ---
# get all valid sizes in range, then subsample ~N_POINTS evenly spaced ones
all_valid = valid_tree_sizes(b, N_MAX)
all_valid = all_valid[all_valid >= N_MIN]

if len(all_valid) <= N_POINTS:
    N_range = all_valid
else:
    # pick N_POINTS indices roughly evenly spaced through the valid sizes
    idx = np.round(np.linspace(0, len(all_valid) - 1, N_POINTS)).astype(int)
    N_range = all_valid[idx]

print(f"N_range ({len(N_range)} sizes): {N_range.tolist()}")

# --- rngs ---
graph_rng = np.random.default_rng(21)
dynamics_rng = np.random.default_rng(42)

# --- output directory ---
save_dir = f'results/C_{C}'
os.makedirs(save_dir, exist_ok=True)

# --- run ---
if weight_dist:
    print(
        f"Simulating for C={C}, weight_dist={weight_dist}, {n_trials} trials")
else:
    print(f"Simulating for C={C}, {n_trials} trials")

for strc in structures:
    results_bag = {}
    print(f"\nStarting structure: {strc}...")

    for N in N_range:
        trials = []

        for t in range(n_trials):
            adj, w = generator(strc, C, int(
                N), weight_dist=weight_dist, rng=graph_rng)
            result = run_trial(adj, w, rng=dynamics_rng, max_steps=50_000)

            trials.append({
                'consensus_time': result['consensus_time'],
                'converged': result['converged'],
                'spectral_gap': result['spectral_gap'],
                'lambda_2_modulus': result['lambda_2_modulus'],
                'predicted_convergence_time': result['predicted_convergence_time'],
                'final_disagreement': result['final_disagreement'],
                'consensus_value': result['consensus_value'],
                'N_actual': adj.shape[0],
            })

        results_bag[int(N)] = trials

        times = [t['consensus_time'] for t in trials]
        converged = sum(t['converged'] for t in trials)
        print(f"  N={N:>6d} | median_t={np.median(times):.0f}, "
              f"converged={converged}/{n_trials}")

    file_path = os.path.join(save_dir, f'{strc}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(results_bag, f)
    print(f"Saved {strc} to {file_path}")

print("\nDone.")
