# Internal consensus under α: slow mode and large N, 2026-11-04

Follow-up to `results/2026-11-04_alpha_sweep_selfloop/`. Two questions:

1. **The ~87-step floor.** Internal consensus time in hierarchies sat at about 87 steps at high α,
   on every structure and N. What sets it?
2. **The crossover.** On growing pyramids at α = 6, the hierarchy stayed flat in N while the
   non-hierarchy slowed; they met at N = 10,000. Does that continue to larger N?

## Configuration

- **Levels:** a deterministic geometric pyramid. Each level holds r times the nodes of the level
  below (r = 0.8 or 0.65).
- **Depth:**
  - Growing: 8/9/10/11/12/13 levels at N = 300/1,000/3,000/10,000/30,000/100,000. That's one level
    per ~3.3x in N, copied from the old tree's median depth up to N = 10,000 and continued at the
    same rate.
  - Fixed: 8 levels at r = 0.8, as a control.
- **Edges:** rule B. Every non-basal node gets exactly 6 subordinates, picked with the level-gap
  kernel. No span-of-control rule: with r = 0.8 each node has about 5 bosses.
- **Influence:** the α rule (`engine.build_weight_matrix`). A junior weights a senior at α, a senior
  weights a junior at 1, and every self-tie is 1. α = 1, 2, 3, 4, 6.
- **Comparison:** q = 0.05 (hierarchy) against q = 1 (non-hierarchy), on the same structure at the
  same N.
- **T:** from the α-sweep calibration, with N = 10,000's value reused above that. Measured q was
  0.040–0.047 and 0.997–1.011 (medians) at every N.
- **Graphs:** 8/8/6/6/4/3 per cell. 210 graphs, no errors.

Code: `alpha_internal/`. Raw rows: `slow_mode.csv`. All tables: `summary_tables.md`.

## 1. The floor is an oscillation, not slow agreement

- In every hierarchy graph at α ≥ 4 (α = 6 for r = 0.65), the eigenvalue that sets internal
  consensus time is negative: λ₂ = −0.823 to −0.857. Opinions flip sign between adjacent levels
  each step, and that oscillation decays only by about 0.85 per step.
- **The mode spans 80–87% of nodes and sits mostly at the bottom.** 31–45% of its weight is on the
  basal level and at most 4% on the top two. So it isn't a top-level wiring detail.
- **λ₂ tends to −0.857 = −6/7 regardless of r, depth, N or α (≥ 4).** That points to the
  6-subordinate rule rather than the pyramid's shape; I haven't derived it.
- **It never occurs at q = 1.** Why it appears in hierarchies: at q ≈ 0 every tie joins adjacent
  levels, so the network is nearly two-colorable (odd vs even levels). At high α each node's
  self-tie is a small share of its row, so synchronous updating nearly alternates. Incoherent ties
  break the alternation.
- This is an artifact of updating everyone at once. In continuous time, or with each node keeping
  at least half its weight on itself, a negative eigenvalue can't slow convergence. The
  **non-oscillating time** (set by the largest real eigenvalue after 1) is what those versions
  would measure, up to a constant factor. It's the cleaner measure, and both are reported below.
  At α ≤ 2 the two are identical everywhere.

## 2. The crossover continues, and widens with N

Hierarchy ÷ non-hierarchy internal consensus time. Below 1 means the hierarchy is faster.

**Non-oscillating time:**

| structure | α | N = 300 | 1,000 | 3,000 | 10,000 | 30,000 | 100,000 |
|---|---|---|---|---|---|---|---|
| growing, r = 0.8 | 1 | 2.49 | 2.56 | 2.62 | 2.68 | 2.71 | 2.72 |
| | 2 | 1.93 | 1.85 | 1.78 | 1.70 | 1.60 | 1.49 |
| | 4 | 1.04 | 0.93 | 0.83 | 0.76 | 0.69 | **0.63** |
| | 6 | 0.78 | 0.70 | 0.63 | 0.58 | 0.53 | **0.49** |
| growing, r = 0.65 | 1 | 3.32 | 3.70 | 4.17 | 4.42 | 4.59 | 4.64 |
| | 2 | 3.71 | 3.77 | 4.06 | 4.28 | 4.33 | 4.37 |
| | 4 | 1.86 | 1.59 | 1.46 | 1.34 | 1.18 | 1.04 |
| | 6 | 1.24 | 1.01 | 0.88 | 0.79 | 0.68 | **0.59** |
| fixed 8 levels, r = 0.8 | 4 | 1.03 | 1.02 | 1.04 | 1.03 | 1.03 | 1.03 |
| | 6 | 0.77 | 0.77 | 0.81 | 0.83 | 0.85 | 0.85 |

**Synchronous time** (largest |eigenvalue|, the measure used in every earlier run):

| structure | α | N = 300 | 1,000 | 3,000 | 10,000 | 30,000 | 100,000 |
|---|---|---|---|---|---|---|---|
| growing, r = 0.8 | 4 | 1.46 | 1.15 | 0.96 | 0.83 | 0.73 | 0.64 |
| | 6 | 1.62 | 1.31 | 1.13 | 1.00 | 0.90 | 0.82 |
| growing, r = 0.65 | 6 | 2.09 | 1.46 | 1.22 | 1.05 | 0.87 | 0.73 |
| fixed 8 levels, r = 0.8 | 6 | 1.61 | 1.56 | 1.58 | 1.56 | 1.55 | 1.55 |

- **With growing depth, whether hierarchy helps or hurts internal consensus depends on α, and the
  answer changes with N.**
  - At α ≤ 2 the hierarchy is slower at every N, 1.5x–4.6x.
  - At α = 4–6 the hierarchy starts slower or about even and becomes faster as N grows. On the
    non-oscillating measure it's about 2x faster by N = 100,000 (r = 0.8, α = 6).
  - The crossover N falls as α rises:
    - r = 0.8, α = 4: between 300 and 1,000
    - r = 0.8, α = 6: below 300
    - r = 0.65, α = 6: about 1,000
    - r = 0.65, α = 4: not yet crossed by 100,000 (ratio 1.04)
  - On the synchronous measure the crossovers come later (3,000–30,000) and the advantage is smaller
    (0.64–0.82 at N = 100,000), because the oscillation sets the hierarchy's floor at 87.
- **With fixed depth the ratio doesn't change with N** for any α. All of the size dependence comes
  from depth growing with N.

**These aren't power laws.** The times grow with depth, and depth grows like ln N, so fitted
exponents shrink as N grows (for example 0.14 → 0.09 for the non-hierarchy at α = 6). Cost per
added level describes it better. Non-oscillating time, r = 0.8, α = 6:

| | 8 levels | 9 | 10 | 11 | 12 | 13 | per added level |
|---|---|---|---|---|---|---|---|
| hierarchy (q = 0.05) | 42.1 | 45.8 | 48.3 | 50.1 | 51.5 | 52.4 | 3.7 → 0.9 (shrinking) |
| non-hierarchy (q = 1) | 53.7 | 65.8 | 76.4 | 86.9 | 97.1 | 106.5 | 12.1 → 9.4 |

At α = 1 it's reversed: each added level costs the hierarchy about 47–58 steps and the
non-hierarchy about 16–21. For r = 0.65 at α = 6, the non-hierarchy's cost per level grows
(15.7 → 18.6) while the hierarchy's shrinks (5.8 → 1.9).

## What this means

- **There's a hierarchy × steepness × size interaction in internal consensus.** When deference is
  shallow, adding levels costs a coherent hierarchy more than an incoherent network. When it's steep,
  adding levels costs the hierarchy almost nothing while the incoherent network keeps paying. So with
  steep deference, hierarchy's advantage grows with group size, and with shallow deference so does its
  disadvantage. It only appears when depth grows with N.
- **Mechanism, not yet checked:** on the growing pyramids with steep deference and coherent levels,
  the slowest non-oscillating mode stays on the bottom levels (52–63% of its weight on the basal
  level) while the levels above copy upward. So added levels barely change the slowest rate. In the incoherent network
  the same deference runs around cycles, and the slow mode spreads.
- **Two modeling choices decide how strong this looks.**
  1. *Synchronous updating with a small self-share* creates the oscillation, which hides half the
     hierarchy's advantage.
  2. *How depth grows with N* sets the rate.

  A self-tie that's a fixed share of each row (at least 1/2) would remove the oscillation. It would
  also fix the stubbornness problems seen with lognormal weights. Both issues point to the same
  design decision.
- Internal consensus was demoted to a regime check in the design pivot. This result is an argument
  for bringing it back as an outcome.
