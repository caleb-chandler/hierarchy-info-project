# α sweep, α·r threshold, and self-tie at 1 (2026-11-04)

Three checks run together on the same graphs. New and self-contained; reuses
`ablation/ablation_model.py`, `ruleB_overnight/model.py` and `engine.build_weight_matrix`
unchanged. Findings: `results/2026-11-04_alpha_sweep_selfloop/SUMMARY.md`.

| File | What it does |
|---|---|
| `sweep_model.py` | The four rule-B structures (fixed 8 levels at r = 0.8; growing depth at r = 0.8, 0.65, 0.5), α-rule influence vector, self-tie rule for lognormal weights |
| `check_sweep.py` | Checks: self-tie 1 via the symmetric route matches the dense W; α-rule influence is stationary and predicts alignment; level sizes; timing |
| `run_sweep.py` | `calibrate` (T for the two new structures; r = 0.8 reuses the ablation's T tables), then `run` |
| `summarize_sweep.py` | Writes `summary_tables.md` in the results folder |
| `weight_tail_check.py` | Why sorted σ = 2 still scales bottom-up with self-tie 1 (growing second moment of the weights) |

Self-tie 1 is W = D⁻¹(A diag(w) + I). It equals D⁻¹K₁ diag(w) with the symmetric
K₁ = A + diag(1/w), so the overnight model's symmetric solvers work with K₁ in place of K.

```
conda run -n diffusion python check_sweep.py            # from alpha_sweep/
conda run -n diffusion python run_sweep.py calibrate    # ~1 min
conda run -n diffusion python run_sweep.py run          # ~9 min
conda run -n diffusion python summarize_sweep.py
conda run -n diffusion python weight_tail_check.py
```
