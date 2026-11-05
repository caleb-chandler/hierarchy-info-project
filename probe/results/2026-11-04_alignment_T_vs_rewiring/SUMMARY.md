# Alignment time: local disorder (T) vs shortcuts (p), 2026-11-04

## Setup

- **Graphs:** the 651 graphs of `results/2026-11-04_internal_T_vs_rewiring/`, rebuilt from the same
  seeds (pairing checked: identical q and node counts for every graph).
- **Structure:** rule B (6 subordinates), r = 0.65, top level held at ≥ 15 nodes (5 → 19 levels over
  N = 300–100,000).
- **Influence:** the α rule, α = 1, 2, 3, 4, 6.
- **Disorder:** T from 0.1 to ∞, or rewiring p from 0 to 1.
- **Sources:** the top or bottom 10% of nodes by trophic level, or a random 10%. Each gives γ = 0.01 of
  its attention to a fixed-opinion environment node.
- **Outcome:** continuous-time alignment time, ln(10⁶)/(1 − ρ).
  - Where 1 − ρ < 10⁻⁹, beyond the eigensolver's precision, the weak-coupling estimate
    ln(10⁶)/(γ·π(S)) is used instead, with π from a power iteration that keeps tiny shares accurate.
    That's 0.4% of rows: bottom-seeded hierarchies at α ≥ 4.
  - Where both are valid they agree: median ratio 0.997, range 0.87–1.00. The largest gaps are
    top-seeded, where the sources hold a large share.

Code: `T_vs_rewiring/` (`run_alignment.py`, `check_alignment.py`, `summarize_alignment.py`,
`make_alignment_figures.py`). Raw rows: `alignment.csv`. Tables for every α: `summary_tables.md`.
Figures: `figures/2026-11-04_alignment_T_vs_rewiring/`.

## Findings

1. **At α = 1, alignment time doesn't depend on N or on structure.** Top seeding takes 10,700–14,300
   steps and bottom seeding 22,900–26,100 for every T and p. Random seeding stays at about 13,800.

2. **Top seeding with α ≥ 2: the coherent hierarchy hits a floor.** As N grows its time falls toward
   ln(10⁶)/γ = 1,382 steps, which is what you get when the top 10% hold nearly all influence. It
   reaches the floor by N ≈ 10,000 at α ≥ 3. The negative exponents (−0.11 to −0.24) reflect this
   saturation, not a power law.
   - Rank-ignoring networks stay flat at 7,300–9,200 steps.
   - At N = 100,000 and α = 6 the hierarchy is 5.3x faster top-down (1,382 vs 7,336).
   - Local disorder at T = 1–2 approaches the floor more slowly.

3. **Bottom seeding with α ≥ 2: the coherent hierarchy matches the α·r prediction.** Each added level
   cuts the bottom's influence by α·r, and the top-size rule adds 1/ln(1/r) levels per unit of ln N.
   So the exponent should approach ln(α·r)/ln(1/r):

   | α | 2 | 3 | 4 | 6 |
   |---|---|---|---|---|
   | measured (T = 0.1) | 0.68 ± 0.02 | 1.57 ± 0.04 | 2.24 ± 0.05 | 3.19 ± 0.07 |
   | prediction | 0.61 | 1.55 | 2.22 | 3.16 |

   Times reach about 10¹⁴ steps at α = 6 and N = 100,000. That's the scaling, not a usable time.

4. **Local disorder reduces the bottom-seeded exponent gradually.** At α = 6:

   | T | 0.5 | 0.75 | 1 | 1.5 | 2 | ≥ 3 |
   |---|---|---|---|---|---|---|
   | exponent | 2.25 | 1.55 | 1.04 | 0.44 | 0.20 | ≈ 0 |

5. **Shortcuts collapse it.** At α = 6:

   | p | 0 | 0.003 | 0.03 | 0.1 | ≥ 0.35 |
   |---|---|---|---|---|---|
   | exponent | 3.18 | 0.42 ± 0.31 | 0.15 | 0.08 | ≈ 0.02 |

   At p = 0.003 only 0.3% of ties are rewired. The same holds at every α ≥ 2.

   **Likely mechanism, not checked directly:**
   - A rewired tie can put a bottom node directly under a high-ranking node, which listens to it at
     weight 1.
   - The high ranks hold nearly all the influence.
   - So a handful of such ties give the bottom a share that no longer shrinks with depth.

   Because a graph has only a few such ties, these cells vary a lot from graph to graph (CIs up to
   ±0.3).

6. **q doesn't determine the outcome.** At similar q the two families differ by an order of
   magnitude. At q ≈ 0.75–0.8 and α = 6, the bottom-seeded exponent is 1.04 for T = 1 against 0.08 for
   p = 0.1. The same split appeared for internal consensus.

7. **Random seeding stays near 13,800 on average, but gets noisy for coherent hierarchies at α ≥ 2.**
   Influence concentrates in the fixed 15-node top tier, and a random 10% includes about 1.5 of those
   nodes on average, sometimes none. Medians at N = 100,000 range from 11,800 to 38,600 with 2 graphs
   per cell. The expected influence share is 10% for any structure, but the typical share isn't once
   influence sits in a fixed-size group.

## Together with internal consensus

In a deep coherent hierarchy, steep deference does three things at once. Each depends on how far ties
reach across ranks:

- **Internal consensus:** faster, with a lower cost per added level, but still slower than a
  rank-ignoring network.
- **Top-down alignment:** as fast as the input allows.
- **Bottom-up alignment:** slower by a factor of α·r per level, which is a power law in N.

Local disorder weakens all three gradually. A small share of shortcuts removes the bottom-up penalty
almost entirely, while barely changing top-down alignment (α = 6, p = 0.003: 1,407 vs 1,382 steps at
N = 100,000).
