# Overnight tests rerun with self-tie 1, 2026-11-04

The four 2026-10-29 overnight tests (`ruleB_overnight/`), rerun with every node's self-tie at
weight 1 instead of its own influence weight w_i. That's the same self-tie the α rule
(`engine.build_weight_matrix`) always used. Results: `results/2026-11-04_ruleB_selftie1/`.
Figures: `figures/2026-11-04_ruleB_selftie1/`.

| File | What it does |
|---|---|
| `run_selftie1.py` | Copy of `ruleB_overnight/run_all.py`; every change is marked "self-tie 1" (`diff` the two files to see them) |
| `compare_to_overnight.py` | Pairs every row with its overnight twin, checks the pairing, writes `comparison_tables.md` |

**The one model change.** Overnight: `W = D⁻¹ K diag(w)`, `K = A + I`, so node i's self-tie is w_i.
Here: `W = D⁻¹ (A diag(w) + I)`, so the self-tie is 1 and `d = A w + 1`. That equals
`D⁻¹ K₁ diag(w)` with the symmetric `K₁ = A + diag(1/w)`, so `ruleB_overnight/model.py`'s
solvers are reused unchanged with K₁ in place of K (checked against a dense build of W in
`alpha_sweep/check_sweep.py`). The stationary influence is still π_i ∝ w_i·d_i, with the new d.

**Paired with the overnight run.** Seeds and the q-calibration tables are unchanged (the tables
were copied, since they depend only on the graphs), so every graph, weight vector and source set
is identical to the overnight run's. With σ = 0 both rules give the same matrix, so those rows
must reproduce the overnight numbers; `compare_to_overnight.py` checks this.

```
conda run -n diffusion python ruleB_selftie1/run_selftie1.py               # from the repo root
conda run -n diffusion python ruleB_overnight/make_figures.py results/2026-11-04_ruleB_selftie1 figures/2026-11-04_ruleB_selftie1
conda run -n diffusion python ruleB_selftie1/compare_to_overnight.py
```
