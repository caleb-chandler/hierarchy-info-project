import os
import pickle
import numpy as np
from generator import create_new, calibrate_density, largest_connected_component
from engine import run_trial
from datetime import date
from pathlib import Path

# --- parameters ---
ALPHA = 2.0                # fixed influence multiplier
B_FRACTION = 0.1           # basal fraction b (B = round(b*N))
DENSITY_MARGIN = 9.0        # safety margin for calibrate_density
N_MIN, N_MAX = 100, 10_000
N_SIZES = 20               # number of sizes in the N-ensemble
SPACING = 'log'             # 'log' or 'lin'
T_HAT_RANGE = np.linspace(0.0, 1.0, 10)
M = 20                       # graph draws per (N, T_hat) cell

# flight check
if any(T_HAT_RANGE) not in range(0.0, 1.0001):
    print("Error: T-hat must stay between 0 and 1")

# --- compute N_range from the chosen spacing ---
if SPACING == 'log':
    N_range = np.unique(np.round(
        np.logspace(np.log10(N_MIN), np.log10(N_MAX), N_SIZES)
    ).astype(int))
elif SPACING == 'lin':
    N_range = np.unique(np.round(
        np.linspace(N_MIN, N_MAX, N_SIZES)
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
run_date = date.today().isoformat()
save_dir = Path(f'results/{run_date}/')
if save_dir.exists():
    count = 1
    for root, dirs, files in os.walk(save_dir):
        for dir_name in dirs:
            count += 1
    save_dir = f'results/{run_date}/{count}'
else:
    save_dir = f'results/{run_date}/1'

os.makedirs(save_dir, exist_ok=True)

# --- run ---
print(f"\nSimulating alpha={ALPHA}, {M} trials per (N, T) cell, "
      f"{len(T_HAT_RANGE)} T values, {len(N_range)} sizes")

for T_hat in T_HAT_RANGE:
    results_bag = {}
    print(f"\nStarting T_hat={T_hat:.3f}...")

    for N in N_range:
        trials = []

        for m in range(M):
            G, T_calibrated, tree_depth, normalized_level_spans, bucket_sizes = create_new(
                N=int(N), b=B_FRACTION, c=c,
                T_hat=T_hat, rng=graph_rng
            )
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
                'T_calibrated': T_calibrated,
                'tree_depth': tree_depth,
                'normalized_level_spans': normalized_level_spans,
                'bucket_sizes': bucket_sizes
            })

        results_bag[int(N)] = trials

        q = [t['trophic_incoherence'] for t in trials]
        conv_times = [t['predicted_convergence_time'] for t in trials]
        print(f"  N={N:>6d} | mean_q={np.mean(q):.4f}  "
              f"median_pred_conv_time={np.median(conv_times):.1f}")

    T_label = f'{T_hat:.2f}'.replace('.', 'p')
    file_path = os.path.join(save_dir, f'T_{T_label}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(results_bag, f)
    print(f"Saved T_hat={T_hat:.3f} to {file_path}")

# --- add md file with params for reference ---

content = f"""
```python
ALPHA : {ALPHA}
B_FRACTION : {B_FRACTION}
DENSITY_MARGIN : {DENSITY_MARGIN}
N_MIN, N_MAX : {N_MIN}, {N_MAX}
N_SIZES : {N_SIZES}
SPACING : {SPACING}
T_HAT_RANGE : {T_HAT_RANGE}
M : {M}
```
"""

with open(os.path.join(save_dir, 'params.md'), 'w', encoding='utf-8') as f:
    f.write(content)

print("\nDone.")
