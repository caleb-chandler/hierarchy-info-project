# Internal-consensus scaling: local disorder (T) vs shortcuts (p), 2026-11-04

## Setup

- **Levels:** geometric pyramid, each level 0.65 times the size of the one below.
- **Depth (top-size rule):** the most levels for which the top level keeps at least 15 nodes. That gives
  5/8/10/13/16/19 levels at N = 300/1,000/3,000/10,000/30,000/100,000.
- **Edges:** rule B (span of control), with exactly 6 subordinates per non-basal node.
- **Influence:** the α rule (self-tie 1), α = 1, 2, 3, 4, 6.
- **Outcome:** continuous-time internal consensus, dx/dt = (W − I)x.
- **Two ways of adding disorder:**
  - **T:** the width of the level-gap kernel. Ties stay near one's own rank (local disorder).
    11 values from 0.1 to ∞.
  - **p:** the share of ties whose lower end is rewired to a random node (shortcuts). 10 values from
    0 to 1.

  p = 0 is the same model as T = 0.1 (the pure hierarchy). p = 1 is the same as T = ∞ (ties ignore rank
  entirely). The endpoints agree within noise.
- **Graphs:** 8/8/6/4/3/2 per cell. 651 graphs, no errors, at most 2.3% of nodes dropped as
  disconnected.

Code: `T_vs_rewiring/`. Raw rows: `scaling.csv`. Full tables: `summary_tables.md`.

**Summary measure.** Time grows roughly linearly with depth, and depth grows like ln N, so fitted
power-law exponents drift with the N range. The main measure here is the **cost per added level**:
the OLS slope of time on depth over every graph, with 95% CIs of ±0.4–1.4 steps. Most cells vary less
than 10% between graphs. Lightly rewired networks (p = 0.003–0.1) and the flat end at N = 100,000 vary
up to 1.5x, so their exponents over N ≥ 10,000 alone are unreliable.

## Cost per added level (steps of internal consensus time)

| T | q | α = 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|
| 0.1 (hierarchy) | 0.00 | 30.0 | 47.7 | 15.3 | 7.4 | 2.7 |
| 0.3 | 0.08 | 29.7 | 48.1 | 15.5 | 7.5 | 2.7 |
| 0.5 | 0.39 | 22.8 | 49.4 | 21.3 | 11.0 | 4.5 |
| 0.75 | 0.60 | 16.2 | 41.4 | **27.6** | 16.7 | 8.2 |
| 1 | 0.74 | 11.2 | 28.8 | **27.7** | **20.4** | 11.7 |
| 1.5 | 0.93 | 5.2 | 12.8 | 17.0 | 17.9 | **15.8** |
| 2 | 1.05 | 2.0 | 5.9 | 8.4 | 10.0 | 11.4 |
| 3 | 1.19 | 1.0 | 1.1 | 1.9 | 2.5 | 3.2 |
| 5 | 1.28 | 1.0 | 0.7 | 0.7 | 0.9 | 1.1 |
| ∞ (flat) | 1.32 | 1.1 | 0.9 | 0.9 | 1.0 | 1.1 |

| p | q | α = 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|---|
| 0 (hierarchy) | 0.00 | 30.0 | 47.8 | 15.3 | 7.4 | 2.7 |
| 0.003 | 0.15 | 28.0 | 41.9 | 13.7 | 8.2 | 4.6 |
| 0.01 | 0.30 | 24.4 | 37.6 | 15.2 | 9.5 | 5.8 |
| 0.03 | 0.47 | 17.1 | 18.1 | 10.2 | 7.8 | 5.8 |
| 0.1 | 0.82 | 7.0 | 7.0 | 4.5 | 3.9 | 3.0 |
| 0.2 | 1.02 | 2.7 | 2.5 | 2.2 | 2.0 | 1.5 |
| 0.35 | 1.16 | 1.3 | 1.1 | 1.1 | 1.1 | 1.0 |
| 1 (flat) | 1.32 | 1.3 | 1.2 | 1.3 | 1.3 | 1.5 |

Time at N = 100,000 relative to the flat network (T = ∞) at the same α:

| | α = 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| hierarchy (T = 0.1) | 6.7 | 12.0 | 4.3 | 2.5 | 1.4 |
| worst local disorder (T = 0.5–1.5) | 5.2 (T = 0.5) | 12.7 (T = 0.5) | 7.9 (T = 1) | 5.8 (T = 1) | 4.8 (T = 1.5) |
| worst rewiring (p > 0) | 6.4 (p = 0.003) | 10.9 (p = 0.003) | 4.7 (p = 0.01) | 3.2 (p = 0.01) | 2.4 (p = 0.03) |

## Findings

1. **A network whose ties ignore rank barely slows with size.** Its cost is about one step per added
   level at every α, and it is the fastest structure at large N in every condition. Depth only matters
   when ties are local in rank.
2. **For the coherent hierarchy, the depth cost falls steeply with steep deference:** 30, 48, 15, 7.4
   and 2.7 steps per level at α = 1, 2, 3, 4, 6. α = 2 is the worst, the same inverted U seen
   before. But the cost doesn't reach zero. At α = 6 the hierarchy is still 1.4x slower than the flat
   network at N = 100,000, and the gap grows with N.
3. **At α ≥ 3, a deep network with local disorder is the worst structure.** Its cost per level peaks at
   intermediate T (q ≈ 0.6–0.95): 28 at α = 3, 20 at α = 4, 16 at α = 6. That's 2–6x the coherent
   hierarchy's cost. At α ≤ 2 the hierarchy itself is at or near the worst.
   - In terms of cost per level against disorder: decreasing at α = 1, peaked at intermediate T for
     α ≥ 3.
4. **Shortcuts are much less harmful than local disorder.** A small rewired fraction removes most of the
   depth cost:
   - at α = 1, p = 0.1 cuts it from 30 to 7 steps per level;
   - by p ≈ 0.35 (14% of ties sideways or up) depth no longer matters at any α.

   At steep α, a few shortcuts do hurt the coherent hierarchy first (α = 6: 2.7 → 5.8 at p = 0.01–0.03).
   But the peak is a third of local disorder's (5.8 vs 15.8).
5. **q alone doesn't determine the outcome.** At similar q, the two families differ several-fold.
   - q ≈ 0.75–0.8: T = 1 costs 11.7 per level at α = 6, against 3.0 for p = 0.1.
   - What matters is how far ties reach across ranks, not how much they deviate from one level down.

## What this changes

- **The earlier hierarchy-beats-non-hierarchy crossover is real but conditional.** The comparison was
  against q = 1 on a local kernel: a deep network with local disorder (finding 3).
  - With steep deference, coherence beats local disorder at scale. At N = 100,000 and α = 6 the
    hierarchy is 3.5x faster than T = 1.5, and the gap widens with depth.
  - It doesn't beat a network whose ties ignore rank, at any α. It doesn't beat substantial rewiring
    either.
- **A version that holds up across both ways of adding disorder:**
  - In a deep structure, every added level costs consensus time in proportion to how local the ties
    are.
  - Coherence plus steep deference cuts that cost.
  - Local disorder in a deep structure is the worst case.
- **The comparison you can defend depends on what's held fixed.** If depth is set by size and span (an
  organization of this size needs this many levels) and the question is whether ties follow the chain
  of command, the T family is the relevant one. Its rank-ignoring end means ties without any rank
  structure.
