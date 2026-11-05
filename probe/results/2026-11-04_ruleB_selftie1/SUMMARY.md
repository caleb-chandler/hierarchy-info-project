# Overnight tests rerun with self-tie 1, 2026-11-04

The four 2026-10-29 overnight tests, unchanged except that every node's self-tie has weight 1
instead of its own influence weight w_i (W = D⁻¹(A diag(w) + I)). This matches the α rule's
self-tie. The run had the same seeds and calibration tables, so every graph, weight vector and
source set is identical to the overnight run's. Run: 17:46–17:59, all 6,000 graphs, no failures.
Code: `ruleB_selftie1/`. Figures: `figures/2026-11-04_ruleB_selftie1/`. Every overnight →
self-tie 1 comparison: `comparison_tables.md`.

**Pairing check.** All 54,600 rows pair one-to-one with the overnight rows (same q, N and
sources). The σ = 0 rows, where both rules give the same matrix, reproduce the overnight times
to within 7 × 10⁻¹².

## Headline

1. **The baseline findings all hold.** Under the baseline weights (sorted, σ = 0.5), alignment
   times in tests 1, 2 and 4 change by at most 12%:
   - alignment time is flat in N;
   - q barely matters;
   - seeding the top aligns 2.7x faster than random and 16x faster than the bottom.
2. **The shuffled-weight size effect is gone.** At σ = 2 the exponents were 0.29–0.53 and are now
   −0.11 to +0.01. It came from heavy nodes ignoring their neighbors, which the old self-tie
   caused.
3. **Sorting still matters a lot, but less than before.** Sorted weights now make top-seeded
   alignment 3x–14x faster than the same weights shuffled; the overnight figure was 3x–140x.
   Part of the old gap was shuffled weights being slowed by stubborn nodes.
4. **Two size effects remain, and neither comes from the hierarchy:**
   - **Bottom seeding with sorted σ ≥ 1.5** (exponent 0.05–0.23 at σ = 2). This comes from the
     heaviest weight growing with N (see the 2026-11-04 α-sweep summary).
   - **Internal consensus under shuffled σ ≥ 1.** The mirror image of the old problem: a node
     whose neighbors are all very light now gives itself up to 99% of its attention.

## Test 1 · Pure hierarchy (q = 0)

| seeding | sorted σ = 0.5: time at N = 10,000 | exponent | equal weights (unchanged) |
|---|---|---|---|
| top 10% | 4,936 → 5,199 | +0.004 | 15,626 |
| random 10% | 14,013 → 13,996 | −0.001 | 13,896 |
| bottom 10% | 94,713 → 83,659 | −0.004 | 24,230 |

Internal consensus is 146 steps at N = 100 and 160 at N = 10,000, against 148 and 162
overnight.

## Test 2 · Random 10% seeding over q = 0.05–1.0 (sorted σ = 0.5)

The exponents are unchanged (−0.02 to +0.01). Time at N = 10,000 across the 10 q values:

| seeding | time |
|---|---|
| random | 13,500–14,000 |
| top | 4,900–5,100 |
| bottom | 82,100–83,800 |

Variation across q is still under 4%.

## Test 3 · Sorted vs shuffled weights

Alignment exponent vs N, range over the 8 q values, overnight → self-tie 1:

| weights | random | top | bottom |
|---|---|---|---|
| sorted σ ≤ 1 | −0.05…+0.01 → −0.04…+0.01 | −0.02…0.00 → −0.02…0.00 | −0.01…+0.04 → −0.01…+0.04 |
| sorted σ = 1.5 | −0.10…−0.01 → −0.08…0.00 | −0.02…−0.01 → −0.03…−0.01 | +0.04…+0.11 → +0.02…+0.11 |
| sorted σ = 2 | −0.11…0.00 → −0.13…−0.03 | −0.02…−0.01 → −0.02…−0.01 | +0.16…+0.26 → **+0.05…+0.23** |
| shuffled σ ≤ 1 | −0.03…+0.01 → −0.03…+0.01 | −0.05…0.00 → −0.04…0.00 | −0.05…+0.02 → −0.04…+0.02 |
| shuffled σ = 1.5 | +0.00…+0.09 → −0.06…+0.01 | −0.13…+0.08 → −0.15…−0.02 | −0.08…0.00 → −0.08…−0.04 |
| shuffled σ = 2 | **+0.29…+0.53** → −0.11…+0.01 | **+0.29…+0.35** → −0.09…−0.06 | **+0.14…+0.29** → −0.14…−0.06 |

Sorted ÷ shuffled alignment time at N = 10,000 (median over q):

| seeding | σ = 0.5 | σ = 1 | σ = 1.5 | σ = 2 |
|---|---|---|---|---|
| top | 0.31 → 0.32 | 0.15 → 0.16 | 0.074 → 0.11 | 0.007 → 0.07 |
| random | 1.0 → 1.0 | 1.0 → 1.0 | 0.79 → 0.99 | 0.11 → 0.90 |
| bottom | 4.0 → 3.4 | 29 → 20 | 295 → 188 | 1,130 → 2,010 |

- **Shuffled weights no longer slow alignment.** At N = 10,000 with shuffled σ = 2, random-seeded
  alignment takes 16,800 steps instead of 164,000. With random seeding, sorting now makes almost
  no difference at any σ (0.90–1.0).
- **Shuffled σ ≥ 1.5 has slightly negative exponents.** This is a sampling effect, not a structural
  one. With heavy-tailed weights, a small source set usually misses the few heavy nodes. At N = 100
  the median influence of a random 10% (10 nodes) is 58% of its mean; at N = 10,000 (1,000 nodes)
  it's 98%. So median time falls as N grows.
- **Sorted weights barely change.** Top-seeded is 1,500–5,000 steps at N = 10,000 and flat in N.
  Bottom-seeded is still extremely slow: 83,000 at σ = 0.5 and 69 million at σ = 2. It still grows
  with N at σ ≥ 1.5, and more at higher q.
- **Internal consensus.** For shuffled weights it is still slow and still grows with N:
  - σ = 1: 0.07–0.29
  - σ = 1.5: 0.31–0.47
  - σ = 2: 0.52–0.68

  It is 4x–33x faster than overnight (median at N = 10,000 for σ = 2: 4,800 instead of
  158,000). The largest self-share is 0.99. For sorted σ = 2, internal consensus takes 550 steps
  (1,685 overnight).
- **The influence-only estimate works again.** For shuffled σ = 2 it is off by more than 20% in
  20% of graphs (74% overnight); for σ = 1.5, in 1% (35%). Everywhere else its median is within
  1.003–1.011 of the exact rate.

## Test 4 · Depth, 4–12 levels (sorted σ = 0.5)

Alignment time at N = 10,000:

| seeding, q | 4 levels | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| top, q = 0.05 (hierarchy) | 7,900 | 6,700 | 5,100 | 4,700 | 4,600 |
| top, q = 1 (non-hierarchy) | 5,400 | 5,200 | 4,900 | 4,600 | 4,400 |
| random, either q | 13,800–14,200 at every depth | | | | |
| bottom, q = 0.05 | 50,900 | 70,600 | 83,700 | 92,500 | 99,000 |
| bottom, q = 1 | 51,500 | 70,000 | 81,200 | 90,200 | 94,400 |

Internal consensus at N = 10,000:

| q | 4 levels | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| 0.05 | 70 | 87 | 159 | 262 | 386 |
| 1.0 | 51 | 49 | 74 | 113 | 159 |

- At every depth alignment is still flat in N (exponents within ±0.035).
- **Adding levels still helps top-down alignment more in hierarchies.** From 4 to 12 levels it gets
  41% faster at q = 0.05, against 19% at q = 1 (overnight: 36% and 17%). A shallow hierarchy aligns
  1.5x slower than a shallow non-hierarchy; by 8 levels they're within 5%.
- Adding levels still roughly doubles bottom-up alignment time, for both.
- **Internal consensus slows with depth, far more in hierarchies:** 5.5x from 4 to 12 levels at
  q = 0.05, against 3.1x at q = 1 (overnight: 7x and 2.8x).

## What this changes

- This is the valid baseline. The overnight conclusions about the baseline weights stand. Only the
  shuffled-weight results and the "stubborn heavy nodes" mechanism in the overnight SUMMARY are
  superseded.
- The original 3x3 study built its matrix with the same self-tie rule (`README.md:112`), so its
  skewed-weight scaling is likely the same artifact.
- Still open:
  - A self-tie as a fixed share of each row would also remove the mirror problem in internal
    consensus.
  - Whether the largest weight should grow with N decides whether sorted heavy-tailed weights
    produce any size effect.
