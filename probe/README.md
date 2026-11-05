# probe/: tests of the model-configuration changes (2026-10-29 onward)

Everything here was built to test changes to the network generator, the influence rule, the update
rule and the outcomes, before porting any of them into the main codebase. Nothing in the repo root
(`engine.py`, `generator.py`, `simulation.py`, notebooks) was modified; the code here imports
`engine.py` and `generator.py` from the repo root unchanged.

Layout: one folder of code per probe, results in `results/`, figures in `figures/`, each named by
date and what was tested. Paths inside these docs are relative to `probe/`, except mentions of
`README.md`, `engine.py` and `generator.py`, which mean the repo root. Run scripts from their own
folder with `conda run -n diffusion python <script>.py`.

## Current configuration (as of 2026-11-04)

- **Network:** geometric level occupancy (each level r = 0.65 times the one below); depth grows by the
  top-size rule (the most levels for which the top level keeps ≥ 15 nodes); rule B, the
  span-of-control rule (every non-basal node has exactly 6 subordinates).
- **Disorder:** the level-gap kernel's width T (local), or a rewiring fraction p (shortcuts).
- **Influence:** the α rule (`engine.build_weight_matrix`): a junior weights a senior at α, the senior
  weights the junior at 1, every self-tie is 1.
- **Update rule:** continuous time (no oscillation artifact; see `alpha_internal`).
- **Outcomes:** internal consensus time; top-, bottom- and random-seeded alignment time (γ = 0.01,
  10% of nodes as sources).

## The probes, in order

| # | Date | Code | Results / figures | What it tested |
|---|---|---|---|---|
| 1 | 10-29 | `ruleB_overnight/` | `results/2026-10-29_ruleB_pyramid_weights/`, `figures/2026-10-29_ruleB_pyramid_weights/` | Rule-B pyramid (8 levels) with sorted lognormal weights in place of α: alignment flat in N, q barely matters. Its shuffled-weight results came from the self-tie rule (superseded by #3). |
| 2 | 10-29 | `ablation/` | `results/2026-10-29_ablation/` | Which of the three overnight changes did what: tree levels drove the old size effect; sorted weights removed the q effect; growing depth + α gives bottom-seeded scaling. |
| 3 | 11-04 | `ruleB_selftie1/` | `results/2026-11-04_ruleB_selftie1/`, `figures/2026-11-04_ruleB_selftie1/` | #1 rerun with self-tie 1 (same graphs): baseline holds; the shuffled-weight size effect was an artifact. Lognormal weights were later dropped for α. |
| 4 | 11-04 | `alpha_sweep/` | `results/2026-11-04_alpha_sweep_selfloop/` | α sweep; the α·r threshold for bottom-seeded alignment (a smooth function of α·r and depth growth); lognormal weights with self-tie 1. |
| 5 | 11-04 | `alpha_internal/` | `results/2026-11-04_alpha_internal_slow_mode/` | Internal consensus under α: the ~87-step floor is a synchronous-update oscillation; continuous time and half-weight updating agree; the hierarchy/non-hierarchy crossover continues to N = 100,000. |
| 6 | 11-04 | `T_vs_rewiring/` | `results/2026-11-04_internal_T_vs_rewiring/`, `results/2026-11-04_alignment_T_vs_rewiring/`, `figures/2026-11-04_internal_T_vs_rewiring/`, `figures/2026-11-04_alignment_T_vs_rewiring/` | Current configuration. Internal consensus and top/bottom/random alignment vs N for local disorder (T) and shortcuts (p), N = 300–100,000, 5–19 levels. |
| 7 | 11-05 | `T_vs_rewiring/` (`run_alpha10.py`) | `results/2026-11-05_internal_alpha10/` | Internal consensus at α = 10 on 217 of #6's graphs: the coherent hierarchy's exponent (0.018) falls below the flat network's (0.061), so it beats flat networks past α ≈ 6–10. |

Each results folder has a `SUMMARY.md` with the findings; each code folder has a `README.md` listing
its scripts and how to rerun them.
