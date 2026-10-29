import os
import pickle
import numpy as np
import generator as gen
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
    'Q_RANGE': np.geomspace(0.05, 1.25, 15),
    'M': 20,
    'SEED': 21
}
P = SimpleNamespace(**params)

# --- rng ---
graph_rng = np.random.default_rng(P.SEED)

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
c = gen.calibrate_density(P.B_FRACTION, P.N_MAX, margin=P.DENSITY_MARGIN)

# Everything above is side-effect free, so `from simulation import P, N_range,
# c` gives any analysis the ground-truth parameters without kicking off a run.
# Everything below only executes under `python simulation.py`.


def main():
    print(
        f"N_range ({len(N_range)} sizes, {P.SPACING}-spaced): {N_range.tolist()}")

    print(f"Q range: {P.Q_RANGE.tolist()}")

    print(
        f"calibrated density c = {c:.4f} (b={P.B_FRACTION}, N_max={P.N_MAX})")

    # --- calibrate the range of T values needed to achieve the desired range of q values ---
    cal_rng = np.random.default_rng(P.SEED + 1)
    T_big = gen.calibrate_T(P.Q_RANGE, 1000, P.B_FRACTION, c, cal_rng)
    T_by_N = {N: gen.calibrate_T(P.Q_RANGE, N, P.B_FRACTION, c, cal_rng) if N < 1000 else T_big
              for N in N_range}

    # T_by_N[N][i] is the T that targets P.Q_RANGE[i] at size N
    T_lines = "\n".join(
        f"{N:>6d} : {np.round(T_by_N[N], 4).tolist()}" for N in N_range)
    print(f"\ncalibrated T (rows N, columns Q_RANGE):\n{T_lines}")

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
    print(f"\nSimulating alpha={P.ALPHA}, {P.M} trials per (N, q) cell, "
          f"{len(P.Q_RANGE)} Q values, {len(N_range)} sizes")

    for i, q in enumerate(P.Q_RANGE):
        results_bag = {}
        print(f"\nStarting q={q:.3f}...")

        for N in N_range:
            trials = []
            T = T_by_N[N][i]

            for _ in range(P.M):
                G, _, tree_depth, normalized_level_spans, bucket_sizes = gen.create_new(
                    N=int(N), b=P.B_FRACTION, c=c,
                    T=T, rng=graph_rng
                )
                Gc, n_dropped = gen.largest_connected_component(G)
                result = run_trial(Gc, alpha=P.ALPHA)

                trials.append({
                    'spectral_gap': result['spectral_gap'],
                    'lambda_2_modulus': result['lambda_2_modulus'],
                    'predicted_convergence_time': result['predicted_convergence_time'],
                    'used_dense_fallback': result.get('used_dense_fallback', False),
                    'actual_q': result['trophic_incoherence'],
                    'mean_trophic_distance': result['mean_trophic_distance'],
                    'N_actual': Gc.number_of_nodes(),
                    'n_dropped': n_dropped,
                    'intended_q': q,
                    'T': T,
                    'tree_depth': tree_depth,
                    'normalized_level_spans': normalized_level_spans,
                    'bucket_sizes': bucket_sizes
                })

            results_bag[int(N)] = trials

            actual_qs = [t['actual_q'] for t in trials]
            conv_times = [t['predicted_convergence_time'] for t in trials]
            print(f"  N={N:>6d} | T={T:.4f}  median_q={np.median(actual_qs):.4f}  "
                  f"median_pred_conv_time={np.median(conv_times):.1f}")

        q_label = f'{q:.4f}'.replace('.', 'p')
        file_path = os.path.join(save_dir, f'q_{q_label}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(results_bag, f)
        print(f"Saved q={q:.3f} to {file_path}")

    # --- add md file with params for reference ---

    content = f"""
```python
ALPHA : {P.ALPHA}
B_FRACTION : {P.B_FRACTION}
DENSITY_MARGIN : {P.DENSITY_MARGIN}
N_MIN, N_MAX : {P.N_MIN}, {P.N_MAX}
N_SIZES : {P.N_SIZES}
SPACING : {P.SPACING}
Q_RANGE : {P.Q_RANGE.tolist()}
M : {P.M}
SEED : {P.SEED} (calibration uses SEED + 1)
```

Calibrated T (rows N, columns Q_RANGE):
```
{T_lines}
```
"""

    with open(os.path.join(save_dir, 'params.md'), 'w', encoding='utf-8') as f:
        f.write(content)

    print("\nDone.")


if __name__ == '__main__':
    main()
