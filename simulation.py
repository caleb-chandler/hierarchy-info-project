import os
import pickle
import numpy as np
from generator import create_new, calibrate_density, largest_connected_component
from engine import run_trial

# --- parameters ---
ALPHA = 2.0                # fixed influence multiplier
B_FRACTION = 0.05           # basal fraction b (B = round(b*N))
DENSITY_MARGIN = 9.0        # safety margin for calibrate_density
N_MIN, N_MAX = 100, 10_000
N_POINTS = 20               # number of sizes in the N-ensemble
SPACING = 'log'             # 'log' or 'lin'
T_RANGE = np.linspace(0.1, 1.0, 10)  # placeholder T sweep
M = 20                       # graph draws per (N, T) cell

# --- compute N_range from the chosen spacing ---
if SPACING == 'log':
    N_range = np.unique(np.round(
        np.logspace(np.log10(N_MIN), np.log10(N_MAX), N_POINTS)
    ).astype(int))
elif SPACING == 'lin':
    N_range = np.unique(np.round(
        np.linspace(N_MIN, N_MAX, N_POINTS)
    ).astype(int))
else:
    raise ValueError(f"Unknown SPACING '{SPACING}'")

print(f"N_range ({len(N_range)} sizes, {SPACING}-spaced): {N_range.tolist()}")

# --- calibrate a single density, held constant across the whole ensemble ---
c = calibrate_density(B_FRACTION, N_MAX, margin=DENSITY_MARGIN)
print(f"calibrated density c = {c:.4f} (b={B_FRACTION}, N_max={N_MAX})")

# --- rng ---
graph_rng = np.random.default_rng(21)

# --- output directory ---
save_dir = 'results/'
os.makedirs(save_dir, exist_ok=True)

# --- run ---
print(f"\nSimulating alpha={ALPHA}, {M} trials per (N, T) cell, "
      f"{len(T_RANGE)} T values, {len(N_range)} sizes")

for T in T_RANGE:
    results_bag = {}
    print(f"\nStarting T={T:.3f}...")

    for N in N_range:
        trials = []

        for m in range(M):
            G = create_new(N=int(N), b=B_FRACTION, c=c, T=T, rng=graph_rng)
            Gc, n_dropped = largest_connected_component(G)
            result = run_trial(Gc, alpha=ALPHA)

            trials.append({
                'spectral_gap': result['spectral_gap'],
                'lambda_2_modulus': result['lambda_2_modulus'],
                'predicted_convergence_time': result['predicted_convergence_time'],
                'used_dense_fallback': result.get('used_dense_fallback', False),
                'trophic_incoherence': result['trophic_incoherence'],
                'mean_trophic_distance': result['mean_trophic_distance'],
                'N_actual': Gc.number_of_nodes(),
                'n_dropped': n_dropped,
            })

        results_bag[int(N)] = trials

        q = [t['trophic_incoherence'] for t in trials]
        conv_times = [t['predicted_convergence_time'] for t in trials]
        print(f"  N={N:>6d} | mean_q={np.mean(q):.4f}  "
              f"median_pred_conv_time={np.median(conv_times):.1f}")

    T_label = f'{T:.3f}'.replace('.', 'p')
    file_path = os.path.join(save_dir, f'T_{T_label}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(results_bag, f)
    print(f"Saved T={T:.3f} to {file_path}")

print("\nDone.")
