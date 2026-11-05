# Overnight tests, 2026-10-29: rule-B pyramid, sorted influence weights, environment input

Everything here is new and self-contained. Nothing in the existing codebase was
modified; `engine.trophic_coherence` and `generator.largest_connected_component`
are imported unchanged. Results are in `results/2026-10-29_ruleB_pyramid_weights/`,
figures and their data tables in `figures/2026-10-29_ruleB_pyramid_weights/`.

| File | What it does |
|---|---|
| `model.py` | Network generator, influence weights, alignment and consensus times, q calibration |
| `check_model.py` | Correctness checks run before the overnight jobs (results below) |
| `run_all.py` | Runs calibration and all four tests in parallel; resumable |
| `make_figures.py` | Figures plus a CSV table behind every figure |
| `diagnose_stubborn_nodes.py` | Diagnostic behind the "mechanism check" table in the results SUMMARY.md |

## The model

**Network (rule B).** Levels 1..D (D = 8 unless stated) form a geometric pyramid:
each level has r = 0.8 times the nodes of the level below, rounded so the counts
sum to N (24%, 19%, 15%, 12%, 10%, 8%, 6%, 5% for 8 levels). Every non-basal node
draws exactly c = 6 distinct subordinates. Each candidate node is weighted by the
kernel exp(-(l_i - l_j - 1)^2 / (2T^2)), so a level's chance is kernel x the nodes
still available in it. T sets q. The largest connected component is kept
(about 0.2% of nodes, mostly basal nodes nobody picked, are dropped).

**Influence weights (α removed).** Ties are undirected with weight 1; every node
also has a tie to itself; everything flowing into node j is scaled by its weight
w_j, then rows are normalized: `W = D^-1 K diag(w)`, `K = undirected adjacency + I`.
This is the construction of the earlier README study. Edge direction is used only
to build the network and compute trophic levels.

Weights are lognormal *quantiles* with median 1: the k-th smallest of n is
`exp(σ · Φ⁻¹((k − 0.5)/n))`. Quantiles instead of random draws keep the largest
weights growing with N exactly as a lognormal sample would, without sampling noise.
- `sorted`: the largest weights go to the highest trophic levels s (ties at random).
- `random`: the same values, shuffled across nodes.
- σ = 0 gives equal weights. The baseline everywhere is sorted, σ = 0.5.

**Environment input.** A node with a fixed opinion; each source gives it a
γ = 0.01 share of its attention and keeps the rest of its row in proportion.
Sources are 10% of nodes: the top or bottom 10% by trophic level s, or a uniformly
random 10%. At q = 0, "bottom 10%" is a random 42% of the basal layer (24% of nodes).

**Outcomes.** Times use the same convention as `engine.compute_spectral_gap`:
ln(10⁻⁶) / ln(rate).
- *Alignment time*: ρ = spectral radius of the real-node block of the matrix with
  the environment node. That block is similar to the symmetric matrix
  `G K G` with `G = diag(sqrt(f·w/d))` (f = 1 − γ on sources, 1 elsewhere), so ρ comes
  from a symmetric eigensolve.
- *Internal consensus time* (no environment): |λ₂| of W, via the symmetric `G K G`
  with f = 1. Recorded as a regime check.
- *Influence of the sources*: because K is symmetric, W is reversible and the
  stationary distribution is exactly π_i ∝ w_i·d_i (d = K w). For weak input,
  1 − ρ ≈ γ·Σ_sources π_i; the CSVs store both so you can see how close it is.

**q calibration.** For each depth and size, the median q of several graphs on a T
grid (0.22–2.8, 63 points) is inverted by linear interpolation. Tables are built per
N below 1,000 and at N = 1,000 for every larger N (the q(T) map stops depending on N
there). Test 1 uses T = 0.1 directly, which puts every edge exactly one level down
(q = 0).

## The four tests

| Test | Varies | Fixed | Repeats |
|---|---|---|---|
| 1 Hierarchy only | N (20 sizes, 100–10,000); seeding top / bottom / random | T = 0.1 (q = 0); weights sorted σ = 0.5, plus equal weights for reference | 20 |
| 2 Random seeding | q (10 values, 0.05–1.0, log-spaced); N (20 sizes) | sorted σ = 0.5; top/bottom also recorded | 10 |
| 3 Influence distribution | q (8 values, 0.05–1.25); N (20 sizes); σ ∈ {0, 0.5, 1, 1.5, 2} sorted; shuffled at σ ∈ {0.5, 1, 1.5, 2} | all weightings on the same graph; seeding top/bottom/random | 10 |
| 4 Depth | levels D ∈ {4, 6, 8, 10, 12}; N ∈ {300, 1,000, 3,000, 10,000}; q (same 10 values as test 2) | r = 0.8 and c = 6; sorted σ = 0.5; seeding top/bottom/random | 5 |

Test 3's "random" condition: your notes asked for one shuffled run against five sorted.
Shuffling at σ = 0.5 (the baseline) is that run; shuffling at σ = 1, 1.5, 2 came for free
on the same graphs and is included as extra columns. In test 4, holding r fixed means
deeper pyramids have a smaller top level (17% of nodes at 4 levels, 1.9% at 12).

Every graph has its own seed `[20261029, test, D, N, q index, repeat]`, so results don't
depend on scheduling, and any single graph can be regenerated.

## Checks run before the overnight jobs (`check_model.py`)

- q(T) at N = 1,000 matches the earlier rule-B prototype (0, ~0.096, ~0.418, ~1.15).
- With equal weights, the weight matrix equals `engine.build_weight_matrix(G, alpha=1)`
  exactly, and internal and alignment times match `engine.compute_spectral_gap` to ~1e-11.
- With unequal weights, ρ and |λ₂| match dense eigensolves of the explicit matrices to
  10 digits, and π = w·d/Σ matches the stationary distribution to 1e-15.
- Sorted weights rise with s (Spearman 0.99); top/bottom sources are the highest/lowest s.

## Output columns (`results/.../test*.csv`, one row per graph x weighting x seeding)

`test, D, N, N_actual, n_dropped, q_index, q_target, T, q, rep, sigma, placement,
internal_time, internal_lam2, seed_rule, n_sources, align_time, align_rate,
approx_rate, source_influence`

- `q` is the measured trophic incoherence of that graph; `q_target` is what T was calibrated for.
- `align_rate` = 1 − ρ; `approx_rate` = γ x `source_influence` (the sources' share of π).

## Rerunning

```
conda run -n diffusion python ruleB_overnight/check_model.py        # from ruleB_overnight/
conda run -n diffusion python ruleB_overnight/run_all.py            # resumes if results exist
conda run -n diffusion python ruleB_overnight/make_figures.py RESULTS_DIR FIGURES_DIR
```
`run_all.py --smoke DIR` runs a tiny version of everything into DIR.
