# Ablation of the 2026-10-29 overnight changes

Switches each of the three overnight changes (levels, edges, weights) between its old and new
form, in every combination. New and self-contained; reuses `generator.create_new`,
`engine.build_weight_matrix`, `engine.trophic_coherence` and `ruleB_overnight/model.py`
unchanged. Findings: `results/2026-10-29_ablation/SUMMARY.md`.

| File | What it does |
|---|---|
| `ablation_model.py` | The structures (tree or pyramid levels x GPPM or 6-subordinate edges) and α-rule solvers |
| `check_ablation.py` | Checks: rule-B copy matches `create_rule_b`; α = 1 matches equal weights; α = 2 matches `engine.compute_spectral_gap` |
| `run_ablation.py` | `calibrate` (T per structure and N), then `run` |
| `summarize_ablation.py` | Exponents, q effects and noise per combination |
| `check_split.py` | Checks for the levels split: tree path reproduces `create_new`; earlier rows unchanged |
| `summarize_split.py` | Levels split into depth / occupancy / basal share (`run_ablation.py ... split`) |
| `gamma_check.py` | Alignment exponents at γ = 0.01–0.9 |

The α rule is not reversible once q > 0, so its times come from general (non-symmetric)
eigensolves rather than the symmetric trick used for node weights.

```
conda run -n diffusion python check_ablation.py           # from ablation/
conda run -n diffusion python run_ablation.py calibrate   # ~2 min
conda run -n diffusion python run_ablation.py run         # ~2.5 min
conda run -n diffusion python summarize_ablation.py
conda run -n diffusion python check_split.py
conda run -n diffusion python run_ablation.py calibrate split
conda run -n diffusion python run_ablation.py run split
conda run -n diffusion python summarize_split.py
conda run -n diffusion python gamma_check.py
```
