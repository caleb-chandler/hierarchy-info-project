# Overnight results, 2026-10-29: rule-B pyramid with sorted influence weights

Run: 03:25–03:49, all 6,000 graphs completed, no failures. Model, parameters and
checks are documented in `ruleB_overnight/README.md`. Figures are in
`figures/2026-10-29_ruleB_pyramid_weights/`; each has a CSV of its numbers in
`tables/`. All times are spectral predictions (steps to reach δ = 10⁻⁶), not
simulated runs.

## Headline

1. **Alignment time does not depend on group size** in any condition except two:
   unequal weights placed at random (shuffled, σ ≥ 1.5), and seeding the bottom
   when unequal weights are sorted by level (σ ≥ 1.5).
2. **Coherence (q) barely matters.** Across a 20-fold range of q, alignment time
   changes by at most a few percent, in every test.
3. **Where influence sits matters enormously.** With the same lognormal weights,
   sorting them by trophic level instead of shuffling them makes top-seeded
   alignment 3x–140x faster and removes the size effect. Shuffled σ = 2 weights give
   exponents of 0.29–0.53 with random or top seeding, close to your earlier README
   study's skewed-weight result (0.41–0.62).
4. **That size effect comes from stubborn heavy nodes.** Because each node's
   self-tie is scaled by its own weight (as in the earlier study), a very heavy
   node surrounded by ordinary neighbors puts 98–99.9% of its attention on itself
   and barely updates. The heaviest weight grows with N, so this bottleneck tightens
   with N. Sorting puts heavy nodes next to other heavy nodes, which prevents it.
   **This is the main design decision to revisit** (see "Decisions" below).

## Test 1 · Pure hierarchy (q = 0), top / bottom / random 10% seeded

`t1_alignment_vs_N.png`, `t1_internal_vs_N.png` · 20 graphs per size, N = 100–10,000

| seeding | sorted σ = 0.5: time at N = 10,000 | exponent | equal weights: time | exponent |
|---|---|---|---|---|
| top 10% | 4,900 | +0.003 | 15,600 | +0.003 |
| random 10% | 14,000 | −0.002 | 13,900 | +0.001 |
| bottom 10% | 94,700 | −0.005 | 24,200 | −0.002 |

- Flat in N for every rule (all exponents within ±0.005).
- With baseline weights, seeding the top aligns 2.8x faster than random and 19x
  faster than the bottom.
- **With equal weights, the top is no better than random** (15,600 vs 13,900).
  Structure alone doesn't make the top influential: top nodes have 6 ties against
  about 11 for middle-level nodes, so their long-run influence is smaller.
- Internal consensus: 147 steps at N = 100, rising to about 160 by N ≈ 700, then
  flat. Alignment is about 30–600x slower, so it reflects where the sources sit.

## Test 2 · Random 10% seeding over q = 0.05–1.0 (sorted σ = 0.5)

`t2_alignment_vs_q.png`, `t2_exponent_vs_q.png` · 10 graphs per cell

- Exponents vs N lie between −0.018 and +0.011 for every q and seeding rule
  (−0.012 to +0.010 fitting N ≥ 300 only).
- At N = 10,000, alignment time across all 10 q values: random 13,500–14,000;
  top 4,650–4,840; bottom 92,900–94,900. That is about 4% variation at most.

## Test 3 · Influence distribution: sorted (σ = 0, 0.5, 1, 1.5, 2) vs shuffled

`t3_exponent_vs_q.png`, `t3_alignment_vs_q.png`, `t3_sorted_vs_shuffled.png` ·
8 q values 0.05–1.25, 20 sizes, 10 graphs per cell; every weighting on the same graphs.
The single shuffled run you asked for is σ = 0.5; σ = 1, 1.5 and 2 shuffled are extras.

Exponent of alignment time vs N (range over the 8 q values):

| seeding | sorted σ ≤ 1 | sorted 1.5 | sorted 2 | shuffled σ ≤ 1 | shuffled 1.5 | shuffled 2 |
|---|---|---|---|---|---|---|
| random | −0.05 to +0.01 | −0.10 to −0.01 | −0.11 to 0.00 | −0.03 to +0.01 | +0.01 to +0.09 | **+0.29 to +0.53** |
| top | −0.02 to 0.00 | −0.02 to −0.01 | −0.02 to −0.01 | −0.05 to 0.00 | −0.13 to +0.08 | **+0.29 to +0.35** |
| bottom | −0.01 to +0.04 | +0.04 to +0.12 | **+0.16 to +0.26** | −0.05 to +0.02 | −0.08 to 0.00 | **+0.14 to +0.29** |

Sorted ÷ shuffled alignment time at N = 10,000 (median over q):

| seeding | σ = 0.5 | σ = 1 | σ = 1.5 | σ = 2 |
|---|---|---|---|---|
| top | 0.31 | 0.15 | 0.074 | 0.007 |
| random | 1.0 | 1.0 | 0.79 | 0.11 |
| bottom | 4.0 | 29 | 295 | 1,100 |

- Sorted weights make top-seeded alignment faster as inequality grows
  (N = 10,000: 15,000 at σ = 0, then 4,700, 2,400, 1,700, 1,450) and keep it flat in N.
- Sorted weights make bottom-seeded alignment extremely slow (94,000 at σ = 0.5,
  600,000 at σ = 1, 7 million at σ = 1.5, 130–210 million at σ = 2). It is the one
  sorted case that grows with N, and it grows faster at higher q (σ = 2: 0.16 at
  q = 0.05 to 0.26 at q = 1.25).
- q makes little difference anywhere else.

**Regime caveat (shuffled σ ≥ 1.5).** There, internal consensus itself is slow
(medians 1,850 steps at σ = 1.5 and 14,500 at σ = 2, against 120–270 for sorted
weights at any σ). Alignment is then only about 3–12x slower than internal consensus, so
those cells measure the group's own mixing speed as well as where the sources sit.
The influence-only estimate is off by more than 20% in 22–81% of those graphs.
Everywhere else its median is within 1.00–1.07 of the exact rate.

**Mechanism check** (diagnostic on fresh graphs, q ≈ 0.16, 3 graphs per size):

| weights | N = 300 → 10,000: internal consensus | largest self-share | nodes in slowest mode |
|---|---|---|---|
| shuffled σ = 2 | 2,000 → 70,000 | 0.979 → 0.999 | 7–28 |
| sorted σ = 2 | 90 → 2,100 | 0.75 → 0.91 | 17–36 |
| shuffled σ = 1 | 360 → 930 | 0.70 → 0.98 | 2–95 |
| sorted σ = 1 | 120 → 130 | 0.43 → 0.56 | 170–1,900 |

Self-share is the share of a node's attention on itself. The slowest mode sits on
a heavy node and the neighbors who copy it.

## Test 4 · Depth, 4–12 levels (r = 0.8, c = 6, sorted σ = 0.5)

`t4_alignment_vs_depth.png`, `t4_exponent_vs_depth.png`, `t4_internal_vs_depth.png` ·
N = 300, 1,000, 3,000, 10,000; 10 q values; 5 graphs per cell

Alignment time at N = 10,000:

| seeding, q | 4 levels | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| top, q = 0.05 (hierarchy) | 7,000 | 6,200 | 4,800 | 4,500 | 4,500 |
| top, q = 1 (non-hierarchy) | 5,000 | 4,900 | 4,600 | 4,400 | 4,200 |
| random, either q | 13,600–14,200 at every depth | | | | |
| bottom, q = 0.05 | 56,600 | 79,300 | 94,700 | 105,200 | 112,800 |
| bottom, q = 1 | 56,900 | 78,600 | 91,800 | 102,500 | 107,400 |

Internal consensus time at N = 10,000:

| q | 4 levels | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| 0.05 | 57 | 83 | 162 | 267 | 396 |
| 1.0 | 59 | 55 | 76 | 116 | 164 |

- At every depth, alignment is flat in N (exponents within about ±0.035), so depth
  changes the constants, not the scaling.
- **Adding levels helps top-down alignment, more for hierarchies.** It's 36% faster
  from 4 to 12 levels at q = 0.05, against 17% at q = 1. A shallow hierarchy aligns
  1.4x slower than a shallow non-hierarchy; by 8 levels they're within 4–6%.
- Adding levels doubles bottom-up alignment time, the same for both.
- **Internal consensus slows with depth, far more in hierarchies:** 7x from 4 to 12
  levels at q = 0.05, against 2.8x at q = 1.

## Data quality

- Every graph completed. No missing or non-finite values. At most 2% of nodes were
  dropped by the largest-component step, at N = 100.
- Measured q came within 0.027–0.040 of every target for N ≥ 300 (median miss
  0.003–0.004), and within 0.05 below N = 300, where small graphs can't produce very
  low q (`tables/q_target_vs_measured.csv`).
- Test 4 uses 5 graphs per cell; graph-to-graph spread is small (see the shaded
  interquartile ranges), but its exponents are the least precise of the four tests.

## Decisions this raises

1. **Self-tie weighting.** Scaling each node's self-tie by its own weight makes heavy
   nodes stubborn, and that alone produces the size effect under shuffled weights.
   Alternatives: keep the self-tie at a fixed weight (e.g. 1, or the median weight),
   or make it a fixed share of each row. Then heavy nodes would be listened to
   without ignoring everyone else. This choice likely decides whether unequal
   influence produces any size effect at all.
2. **Where the hierarchy effect lives.** In this model, coherence (q) has almost no
   effect, while placing influence by rank has a large one. That holds even at
   q = 1.25, where levels carry little structure, so the benefit of sorting may come
   from heavy nodes being tied to other heavy nodes rather than from coherence itself.
   That's an open question; it could be tested by sorting weights by something other
   than trophic level.
3. **Bottom-up information.** Sorting makes seeding the base extremely slow at high
   inequality, and that slowdown grows with N and q. If the theory is about
   front-line information reaching leaders, this is the condition where size and
   structure both matter.
