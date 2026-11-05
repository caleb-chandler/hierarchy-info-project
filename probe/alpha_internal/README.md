# Internal consensus under α: slow mode and large N (2026-11-04)

Follow-up to `alpha_sweep/`. Finds what sets the ~87-step internal-consensus floor in hierarchies at
high α, and extends the growing-pyramid comparison to N = 100,000. New and self-contained; reuses
`alpha_sweep/sweep_model.py`, `ablation/ablation_model.py` and `engine.build_weight_matrix`
unchanged. Findings: `results/2026-11-04_alpha_internal_slow_mode/SUMMARY.md`.

| File | What it does |
|---|---|
| `run_internal.py` | Per graph and α: λ₂ (largest \|eigenvalue\| after 1) and its eigenvector's spread over levels, plus the same for the slowest non-oscillating mode (largest real eigenvalue after 1) |
| `check_internal.py` | Checks: λ₂ time matches `ablation_model.internal_alpha`; eigenvector residual |
| `summarize_internal.py` | Writes `summary_tables.md` in the results folder |
| `update_rule_check.py` | Synchronous vs continuous-time vs half-weight times from the stored eigenvalues, and power law vs log law; writes `update_rule_tables.md` |

Growing depth continues the alpha-sweep schedule at the same rate (12 levels at N = 30,000, 13 at
100,000). T for N > 10,000 reuses the N = 10,000 value; the measured q of every graph is in the CSV
and in the first table of `summary_tables.md`. Runs 4 workers, since each uses about 1 GB at
N = 100,000.

```
conda run -n diffusion python check_internal.py       # from alpha_internal/
conda run -n diffusion python run_internal.py         # ~9 min
conda run -n diffusion python summarize_internal.py
```
