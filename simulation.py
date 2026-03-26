import os
import numpy as np
from generator import generator
from engine import run_trial
import pickle

# --- parameters ---
C = 6
N_range = np.arange(10, 10011, 500)  # 21 sizes from 10 to 10010
weight_dist = None
n_trials = 20  # trials per (structure, N) — graph is regenerated each trial

structures = ['control', 'alternative', 'hierarchy']

# separate rngs so graph generation and dynamics are independently seeded
graph_rng = np.random.default_rng(21)
dynamics_rng = np.random.default_rng(42)

# --- ensure directory exists ---
save_dir = f'results/C_{C}'
os.makedirs(save_dir, exist_ok=True)

# --- simulation loop ---
if weight_dist:
    print(f"Simulating for C={C}, weight_dist={weight_dist}, {n_trials} trials per condition")
else:
    print(f"Simulating for C={C}, uniform weights, {n_trials} trials per condition")

for strc in structures:
    results_bag = {}

    print(f"\nStarting structure: {strc}...")

    for N in N_range:
        trials = []

        for t in range(n_trials):
            # generate a fresh graph each trial
            adj, w = generator(strc, C, int(N), weight_dist=weight_dist, rng=graph_rng)

            # run dynamics
            result = run_trial(adj, w, rng=dynamics_rng)

            # store only what we need for analysis
            trials.append({
                'consensus_time': result['consensus_time'],
                'converged': result['converged'],
                'spectral_gap': result['spectral_gap'],
                'lambda_2_modulus': result['lambda_2_modulus'],
                'predicted_convergence_time': result['predicted_convergence_time'],
                'final_disagreement': result['final_disagreement'],
                'consensus_value': result['consensus_value'],
                'N_actual': adj.shape[0],  # may differ from N for hierarchy
            })

        results_bag[int(N)] = trials
        
        # summary for monitoring
        times = [t['consensus_time'] for t in trials]
        converged = sum(t['converged'] for t in trials)
        print(f"  N={N:>6d} (actual={trials[0]['N_actual']:>6d}) | "
              f"median_t={np.median(times):.0f}, "
              f"converged={converged}/{n_trials}")

    # save
    file_path = os.path.join(save_dir, f'{strc}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(results_bag, f)

    print(f"Saved {strc} to {file_path}")

print("\nDone.")