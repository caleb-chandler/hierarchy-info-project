# α sweep, α·r threshold, and self-tie at 1, 2026-11-04

Three checks on the same graphs:

1. **Threshold.** Does bottom-seeded alignment get a size effect only when α·r > 1? Tested on
   growing pyramids with r = 0.8, 0.65 and 0.5, so α·r = 1 falls at α = 1.25, 1.54 and 2.
2. **α sweep.** α = 1, 1.25, 1.6, 2, 2.5, 3, 4, 6 on every structure. α = 2 was arbitrary.
3. **Self-tie at 1.** Lognormal weights (σ = 0–2, sorted or shuffled), with the self-tie scaled by the
   node's own weight (the overnight rule, "own") or fixed at 1 ("one"). Every α run already had a
   self-tie of 1.

Setup: rule B with 6 subordinates on every structure. Fixed 8 levels at r = 0.8 (the "new" structure).
Growing depth: 8/9/10/11 levels at N = 300/1,000/3,000/10,000 for r = 0.8 and 0.65, and 7/8/9/10 for
r = 0.5. Every growing structure adds 3 levels over the range. q = 0.05, 0.3, 1.0, but r = 0.5 can't
reach q = 1 (its maximum is about 0.92). 16/16/12/10 graphs per cell. γ = 0.01, with 10% of nodes as
sources. 594 graphs, no errors. At most 4.7% of nodes were dropped as disconnected (r = 0.5, N = 10,000:
basal nodes left with no boss). Code: `alpha_sweep/`. Full tables: `summary_tables.md`. Raw rows:
`alpha.csv`, `lognormal.csv`.

## 1. The α·r threshold

Bottom-seeded alignment, exponent vs N at q = 0.05. In parentheses: the "level formula," which has no
fitted inputs. It is the slope of 1 / (bottom level's share) when level l holds influence k_l·α^l.

| α | r = 0.8: α·r | measured (formula) | r = 0.65: α·r | measured (formula) | r = 0.5: α·r | measured (formula) |
|---|---|---|---|---|---|---|
| 1 | 0.80 | +0.02 (+0.03) | 0.65 | +0.00 (+0.01) | 0.50 | −0.00 (+0.00) |
| 1.25 | 1.00 | +0.09 (+0.09) | 0.81 | +0.03 (+0.03) | 0.62 | +0.01 (+0.01) |
| 1.6 | 1.28 | +0.23 (+0.24) | 1.04 | +0.11 (+0.11) | 0.80 | +0.04 (+0.04) |
| 2 | 1.60 | +0.40 (+0.41) | 1.30 | +0.25 (+0.25) | 1.00 | +0.12 (+0.11) |
| 2.5 | 2.00 | +0.58 (+0.60) | 1.62 | +0.42 (+0.43) | 1.25 | +0.25 (+0.24) |
| 3 | 2.40 | +0.74 (+0.75) | 1.95 | +0.57 (+0.58) | 1.50 | +0.39 (+0.38) |
| 4 | 3.20 | +0.99 (+1.00) | 2.60 | +0.82 (+0.83) | 2.00 | +0.64 (+0.62) |
| 6 | 4.80 | +1.34 (+1.35) | 3.90 | +1.17 (+1.19) | 3.00 | +1.01 (+0.98) |

95% CIs are ±0.01–0.02 throughout.

- **The exponent depends on α·r, not on α or r separately.** For example, at α·r ≈ 1.6 the three
  structures give 0.40, 0.42 and 0.39 (r = 0.5 at α·r = 1.5). The level formula matches every cell
  within 0.03.
- **The mechanism is confirmed.** Per member, each level is exactly α times as influential as the level
  below (measured 1.00, 1.25, 1.60, 2.00, 2.49, 2.99, 3.98, 5.95 at every r).
- **The threshold is a smooth crossover at these depths, not a switch.** Up to α·r ≈ 0.65 the exponent
  is about 0. At 0.8 it's 0.02–0.04, and at 1 it's 0.09–0.12. Above 1 it's close to
  (levels added per ln N) × ln(α·r) = 0.86 ln(α·r). Strictly, "zero below 1" only holds as depth goes
  to infinity.
- **Incoherence weakens it.** At q = 0.3 the exponents are 70–85% of the q = 0.05 values. At q = 1 they
  are about 25% (r = 0.8) and about 11% (r = 0.65).
- **Fixed depth gives no size effect at any α or q** (every exponent between 0.00 and 0.01).
- The measured exponent equals the exponent implied by the sources' influence share, 1 / π(S), in
  every cell. So the size effect comes entirely from influence shifting toward the top as N grows.

## 2. α sweep

Fixed 8 levels, r = 0.8. Median time at N = 10,000. "Ratio" is hierarchy ÷ non-hierarchy
(q = 0.05 ÷ q = 1); below 1 means the hierarchy is faster.

| α | top ratio | bottom ratio | random (q = 0.05 / 1) | internal ratio |
|---|---|---|---|---|
| 1 | 1.04 | 0.95 | 13,720 / 13,927 | 2.47 |
| 1.25 | 0.71 | 1.35 | 13,770 / 13,920 | 2.51 |
| 1.6 | 0.55 | 2.40 | 13,853 / 13,927 | 2.27 |
| 2 | 0.51 | 4.66 | 13,976 / 13,936 | 1.91 |
| 3 | 0.52 | 19.3 | 14,064 / 13,986 | 1.32 |
| 4 | 0.56 | 57.7 | 14,022 / 14,064 | 1.41 |
| 6 | 0.62 | 287 | 13,938 / 14,061 | 1.56 |

- **At α = 1, hierarchy and non-hierarchy align the same** (1.04 and 0.95). The whole q effect on
  alignment comes from α.
- **The top-seeded advantage levels off.** It reaches about 2x by α = 2 and shrinks to 0.62 at α = 6,
  because non-hierarchies also speed up as α rises. The bottom-seeded penalty keeps growing steeply.
  On the growing r = 0.8 pyramid it runs from 1.60 (α = 1.25) to 7,900 (α = 6).
- **Random seeding stays at 13,300–14,100 for every α on both r = 0.8 structures**, as expected.
- **No size effect at fixed depth for any α** (exponents within ±0.02). The one exception is random
  seeding at q = 0.05 with α ≥ 3, which comes out at −0.06 to −0.10 ± 0.06–0.08. That's noise from
  influence concentrating at the top, so it matters a lot whether a random 10% happens to include
  top nodes.
- **Internal consensus.** The hierarchy is 2.5x slower at α = 1 and 1.3–1.6x slower at α ≥ 3. On the
  growing r = 0.8 pyramid, the internal-consensus exponent from depth growth (about 0.2 at α ≤ 2) drops
  to 0.01 at α ≥ 4 when q = 0.05. With steep deference, the top sets the consensus and depth stops
  mattering. At q = 1 it stays at 0.13–0.19.

## 3. Self-tie at 1 vs scaled by the node's own weight

Fixed 8 levels, r = 0.8. Exponent vs N, range over q = 0.05 / 0.3 / 1, own → one:

| weights | top | random | bottom | internal |
|---|---|---|---|---|
| shuffled σ = 1 | −0.04…+0.00 → −0.04…+0.01 | −0.04…+0.02 → −0.02…+0.01 | −0.01…+0.03 → −0.01…+0.02 | +0.38…+0.45 → +0.09…+0.20 |
| shuffled σ = 1.5 | −0.07…+0.21 → −0.09…+0.02 | +0.01…+0.10 → −0.04…+0.02 | −0.06…+0.05 → −0.05…−0.02 | +0.62…+0.93 → +0.33…+0.47 |
| shuffled σ = 2 | +0.18…+0.61 → −0.27…+0.06 | +0.50…+0.57 → −0.01…+0.01 | +0.24…+0.37 → −0.05…−0.01 | +1.05…+1.17 → +0.51…+0.65 |
| sorted σ = 1.5 | −0.02…−0.01 → −0.02…−0.00 | −0.11…+0.02 → −0.10…+0.01 | +0.04…+0.09 → +0.01…+0.08 | +0.41…+0.48 → +0.00…+0.32 |
| sorted σ = 2 | −0.01…−0.01 → −0.02…−0.00 | −0.14…+0.02 → −0.14…−0.01 | +0.15…+0.20 → +0.04…+0.17 | +0.65…+0.91 → +0.00…+0.54 |

- **With a self-tie of 1, shuffled weights no longer give alignment a size effect.** Random seeding goes
  from 0.50–0.57 to −0.01–0.01, and at N = 10,000, q = 0.05 its time drops from 133,000 to 16,300. The
  overnight run's shuffled-weight result was the stubborn-node artifact. The original 3x3 study built
  its matrix the same way (`README.md:112`: self-loops added first, then columns scaled by weight), so
  its skewed-weight scaling may be the same artifact. That wasn't tested here.
- The influence-only estimate also works again. For shuffled σ = 2, the share of graphs where it's off
  by more than 20% falls from 74% to 7%.
- **Internal consensus still scales under shuffled heavy-tailed weights** (σ = 2: 0.51–0.65). A self-tie
  of 1 creates the mirror-image problem: a node whose neighbors are all very light now puts up to 98.7%
  of its attention on itself, and the lightest weights get lighter as N grows. Making the self-tie a
  fixed share of each row would avoid both versions of the problem. This doesn't affect alignment at
  γ = 0.01, but it would at strong coupling.
- **Sorted σ = 2 with bottom seeding still scales** (0.04–0.17). This comes from the weights, not the
  structure. The alignment exponent equals the exponent of 1 / π(bottom 10%) exactly. Sorting ties heavy
  nodes to heavy nodes, so the top's influence (π_i ∝ w_i·d_i) grows roughly like w². The mean of w²
  over the quantile grid keeps growing across N = 300–10,000 (slope 0.25 at σ = 2, 0.10 at σ = 1.5)
  because the largest weight grows with N. It's a consequence of letting the top weight grow with
  group size (`alpha_sweep/weight_tail_check.py`).
- Sorted weights at σ ≤ 1 are almost unchanged by the self-tie rule.
- On the growing r = 0.8 pyramid the pattern is mostly the same. Under shuffled σ = 2, random seeding
  goes from 0.36–0.59 to −0.02…+0.16, so less of the effect disappears there. Under sorting, the
  bottom-seeded exponent rises with σ (0.05 at σ = 0.5, then 0.28–0.35 at σ = 2) for both rules.

## What this means

- Every size effect in alignment that survives the self-tie fix comes from the same channel: the
  sources' share of influence drifts as N grows. That drift has two causes. One is depth growing with
  N while each level holds α·r times the influence of the level below. The other is sorted weights
  whose maximum grows with N. Random seeding has no size effect in any condition.
- On your "degree, not kind" point: at realistic depths the exponent is a smooth function of α·r and
  of how fast depth grows. You were mostly right. The difference in kind is between fixed depth (no
  effect at any α) and growing depth.
- Open choices: the self-tie as a fixed share of each row; whether the largest weight should grow
  with N; and which values of α and r are empirically plausible for the hierarchies the paper is about.
