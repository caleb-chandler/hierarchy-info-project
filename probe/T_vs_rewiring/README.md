# Internal-consensus scaling: kernel width T vs rewiring fraction p (2026-11-04)

Continuous-time internal consensus vs N on rule-B pyramids whose depth follows the top-size rule
(r = 0.65, top level held at ≥ 15 nodes: 5 → 19 levels over N = 300–100,000). Disorder comes
either from the level-gap kernel's width T (local: ties stay near one's own rank) or from a
rewiring fraction p (shortcuts: each tie's lower end goes to a random node with probability p).
p = 0 is the same model as T = 0.1, and p = 1 the same as T = ∞. New and self-contained; reuses
`ablation/ablation_model.py`, `ruleB_overnight/model.py` and `engine.build_weight_matrix` unchanged.
Findings: `results/2026-11-04_internal_T_vs_rewiring/SUMMARY.md`.

| File | What it does |
|---|---|
| `scaling_model.py` | Top-size depth rule, the rewired rule-B generator, tie statistics, continuous-time solver |
| `check_scaling.py` | Checks: depth schedule; shared endpoints of the two families; continuous time vs dense eigensolve; timing |
| `run_scaling.py` | Every graph under α = 1, 2, 3, 4, 6; appends rows as graphs finish and resumes on rerun |
| `summarize_scaling.py` | Writes `summary_tables.md` in the results folder |
| `make_figures.py` | Figure 1 (time vs N at α = `FIG1_ALPHA`, currently 2), figure 2 (exponent vs α) and figure 5 (the same exponents vs T / p, one line per α), T and p side by side, into `figures/2026-11-04_internal_T_vs_rewiring/` with CSVs in `tables/` |
| `check_alignment.py` | Checks: power-iteration influence vs the eigensolver; weak-coupling estimate vs exact alignment time; timing |
| `run_alignment.py` | Top-, bottom- and random-seeded alignment on the same graphs (same seeds); exact time where 1 − ρ > 10⁻⁹, else the weak-coupling estimate |
| `summarize_alignment.py` | Pairing check against `scaling.csv` and tables, into `results/2026-11-04_alignment_T_vs_rewiring/summary_tables.md` |
| `make_alignment_figures.py` | Figure 3 (alignment time vs N at α = 3) and figure 4 (alignment exponent vs α, with the α·r prediction), rows top / bottom seeded, into `figures/2026-11-04_alignment_T_vs_rewiring/` |
| `fig_style.py` | Shared figure style and helpers |

```
conda run -n diffusion python check_scaling.py       # from T_vs_rewiring/
conda run -n diffusion python run_scaling.py         # ~20 min on 8 workers
conda run -n diffusion python summarize_scaling.py
conda run -n diffusion python make_figures.py
conda run -n diffusion python check_alignment.py
conda run -n diffusion python run_alignment.py       # ~13 min on 8 workers
conda run -n diffusion python summarize_alignment.py
conda run -n diffusion python make_alignment_figures.py
```
