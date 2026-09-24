import os
import pickle
import numpy as np
from generator import create_new, calibrate_density, largest_connected_component
from engine import run_trial
from datetime import date
from pathlib import Path
from types import SimpleNamespace

# --- parameters ---
params = {
    'ALPHA': 2.0,
    'B_FRACTION': 0.1,
    'DENSITY_MARGIN': 9.0,
    'N_MIN': 100,
    'N_MAX': 10_000,
    'N_SIZES': 20,
    'SPACING': 'log',
    'T_HAT_RANGE': np.linspace(0.1, 1.0, 10),
    'M': 20
}
P = SimpleNamespace(**params)

# flight check
if not np.all((P.T_HAT_RANGE >= 0.0) & (P.T_HAT_RANGE <= 1.0)):
    print("Error: T-hat must stay between 0 and 1")

# --- compute N_range from the chosen spacing ---
if P.SPACING == 'log':
    N_range = np.unique(np.round(
        np.logspace(np.log10(P.N_MIN), np.log10(P.N_MAX), P.N_SIZES)
    ).astype(int))
elif P.SPACING == 'lin':
    N_range = np.unique(np.round(
        np.linspace(P.N_MIN, P.N_MAX, P.N_SIZES)
    ).astype(int))
else:
    raise ValueError(f"Unknown SPACING '{P.SPACING}'")

# --- calibrate a single density, held constant across the whole ensemble ---
c = calibrate_density(P.B_FRACTION, P.N_MAX, margin=P.DENSITY_MARGIN)

# Everything above is side-effect free, so `from simulation import P, N_range,
# c` gives any analysis the ground-truth parameters without kicking off a run.
# Everything below only executes under `python simulation.py`.


def main():
    print(
        f"N_range ({len(N_range)} sizes, {P.SPACING}-spaced): {N_range.tolist()}")

    print(f"T-hat range: {P.T_HAT_RANGE.tolist()}")

    print(
        f"calibrated density c = {c:.4f} (b={P.B_FRACTION}, N_max={P.N_MAX})")

    # --- rng ---
    graph_rng = np.random.default_rng(21)

    # --- output directory ---
    run_date = date.today().isoformat()
    save_dir = Path(f'results/{run_date}/')
    count = 0
    if save_dir.exists():
        for _, dirs, _ in os.walk(save_dir):
            for _ in dirs:
                count += 1
        save_dir = f'results/{run_date}/{count}'

    os.makedirs(save_dir, exist_ok=True)

    # --- run ---
    print(f"\nSimulating alpha={P.ALPHA}, {P.M} trials per (N, T) cell, "
          f"{len(P.T_HAT_RANGE)} T values, {len(N_range)} sizes")

    for T_hat in P.T_HAT_RANGE:
        results_bag = {}
        print(f"\nStarting T_hat={T_hat:.3f}...")

        for N in N_range:
            trials = []

            for m in range(P.M):
                G, T_calibrated, tree_depth, normalized_level_spans, bucket_sizes = create_new(
                    N=int(N), b=P.B_FRACTION, c=c,
                    T_hat=T_hat, rng=graph_rng
                )
                Gc, n_dropped = largest_connected_component(G)
                result = run_trial(Gc, alpha=P.ALPHA)

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
ALPHA : {P.ALPHA}
B_FRACTION : {P.B_FRACTION}
DENSITY_MARGIN : {P.DENSITY_MARGIN}
N_MIN, N_MAX : {P.N_MIN}, {P.N_MAX}
N_SIZES : {P.N_SIZES}
SPACING : {P.SPACING}
T_HAT_RANGE : {P.T_HAT_RANGE}
M : {P.M}
```
"""

    with open(os.path.join(save_dir, 'params.md'), 'w', encoding='utf-8') as f:
        f.write(content)

    print("\nDone.")


if __name__ == '__main__':
    main()
