# Ablation of the overnight changes, 2026-10-29

The overnight run changed three things at once. Here each is switched between its old and
new form on its own:

| change | old | new |
|---|---|---|
| levels | GPPM uniform-attachment tree, b = 0.1 (random depth: median 8 levels at N = 300, 10.5–11 at 10,000) | 8-level pyramid, r = 0.8 |
| edges | 1 tree subordinate + kernel excess edges, 7.12 edges per node | exactly 6 subordinates per non-basal node, 4.56 edges per node |
| weights | α = 2 (junior weights senior x2, self-tie 1) | lognormal σ = 0.5 sorted by trophic level |

With three changes, "undo one" and "keep only one" together are the full 2x2x2 factorial.
Extras: equal weights on every structure, and GPPM edges thinned to rule B's edge count.

Setup: N = 300, 1,000, 3,000, 10,000; q = 0.05, 0.3, 1.0 (T calibrated per structure and N;
every target reached); 16/16/12/10 graphs per cell; every graph evaluated under all three
weightings with the same sources (10% of nodes, γ = 0.01). Code and checks: `ablation/`.
Numbers below: `summary_long.csv` (all combinations x outcomes), raw rows `ablation.csv`.

## Internal consensus (the original headline outcome)

| combination | exponent vs N (over the 3 q) | hierarchy ÷ non-hierarchy at N = 10,000 | graph-to-graph noise (SD of log10) |
|---|---|---|---|
| OLD | 0.56 – 0.68 | 1.11 | 0.31 |
| old + pyramid levels only | 0.01 | 2.32 | 0.008 |
| old + 6-subordinate edges only | 0.05 – 0.16 | 2.50 | 0.09 |
| old + sorted weights only | 0.48 – 0.58 | 1.14 | 0.24 |
| NEW | 0.01 – 0.02 | 2.09 | 0.005 |
| new with tree levels back | 0.06 – 0.13 | 1.74 | 0.06 |
| new with GPPM edges back | 0.01 – 0.02 | 2.22 | 0.013 |
| new with α = 2 back | 0.01 | 1.91 | 0.005 |

- The old size effect came from the level structure, not the weights. Pyramid levels alone
  remove it, and the noise with it.
- 6-subordinate edges alone remove most of it. The rest (0.12–0.16 at q ≤ 0.3, 0.05 at q = 1)
  follows depth, which still grows with N under tree levels. It is larger for hierarchies.
- The new model has a *larger* q effect on internal consensus than the old one: depth noise
  was hiding it.
- Equal weights on the old structure give 0.31–0.47, so α = 2 amplified the old exponents.

## Alignment with the environment

No combination gives alignment a size effect: every exponent is within ±0.065.

Random seeding is fixed by construction: 13,710–14,020 steps at N = 10,000 in all 15
combinations, against ln(10⁶) / (γ x 0.10) = 13,816. A random 10% holds 10% of the influence
whatever the structure or weights.

Hierarchy ÷ non-hierarchy alignment time at N = 10,000 (below 1: the hierarchy is faster):

| combination | top seeded | bottom seeded |
|---|---|---|
| OLD | 0.61 | 1.75 |
| old + pyramid levels only | 0.44 | 2.60 |
| old + 6-subordinate edges only | 0.50 | 1.35 |
| old + sorted weights only | 1.08 | 0.91 |
| NEW | 1.04 | 1.02 |
| new with tree levels back | 0.90 | 0.63 |
| new with GPPM edges back | 1.11 | 0.74 |
| new with α = 2 back | 0.50 | 4.65 |

- **Replacing α with sorted weights is what removed the q effect.** It does so on the old
  model too.
- With α = 2 on the new structure: top-seeded 3,110 steps at q = 0.05 against 6,180 at q = 1;
  bottom-seeded 280,000 against 60,200. Noise stays low (0.005 and 0.013).
- Why: sorted weights give the highest trophic levels the heaviest weights at every q, so the
  influence ranking is imposed whatever the structure. Under α, influence comes from tie
  direction, so it follows levels only as far as the edges do, and fades as q rises.
- Equal weights show no q effect on top-seeded alignment (1.04 on the new structure).

## Splitting the levels change (`ablation_split.csv`, `summary_split.csv`)

The levels factor above changed depth, occupancy shape, occupancy randomness and basal share
together. Three more level types, run on the same grid with both edge rules and all weightings:

| level type | depth (N = 300 → 10,000) | occupancy | basal |
|---|---|---|---|
| tree (old) | 8 → 11, random | tree-shaped, random; top levels ~18, 4, 1 nodes at N = 10,000 | 10% |
| tree capped at 8 | 8 | tree-shaped, random; capping also fills the top (591, 244, 76) | 10% |
| tree capped at 8, 24% basal | 6 → 8 | tree-shaped, random; top still sparse (93, 10, 1) | 24% |
| growing pyramid | 8, 9, 10, 11 (the tree's median) | geometric r = 0.8, fixed | 24% → 22% |
| pyramid (new) | 8 | geometric r = 0.8, fixed | 24% |

**Internal consensus**, old edges and α = 2 (exponent vs N; noise = SD of log10):

| level type | exponent | noise |
|---|---|---|
| tree | 0.56 – 0.68 | 0.31 |
| tree capped at 8 | 0.15 – 0.23 | 0.12 |
| growing pyramid | 0.21 – 0.26 | 0.01 |
| pyramid | 0.01 | 0.008 |

- Depth growth alone (growing pyramid vs pyramid) gives an exponent of about 0.2, with no
  noise. It does so under every edge rule and weighting (0.16–0.26) and about equally at
  q = 0.05 and q = 1.
- The tree's occupancy (sparse top) adds roughly 0.35–0.4 on top of that with GPPM edges, plus
  all of the noise. With 6-subordinate edges it adds nothing: the tree's extra levels hold so
  few nodes that, once each is tied in by 6 edges, they barely lengthen the hierarchy
  (tree 0.05–0.16 against the growing pyramid's 0.16–0.20).
- The 24%-basal capped tree is not a clean test of basal share: the cap rarely binds, depth
  still grows from 6 to 8 and the top stays sparse, so it behaves like the uncapped tree.

**Alignment: depth growth plus α gives a size effect, and it depends on coherence.**
Growing pyramid, α = 2 (exponent [95% CI]; median time at N = 10,000):

| edges | seeding | q = 0.05 | q = 0.3 | q = 1 |
|---|---|---|---|---|
| 6 subordinates | bottom | +0.39 [0.38, 0.40]; 1,074,000 | +0.33; 631,000 | +0.10; 84,000 |
| 6 subordinates | top | −0.13; 1,960 | −0.13; 2,130 | −0.10; 4,340 |
| GPPM | bottom | +0.27; 321,000 | +0.23; 218,000 | +0.06; 58,000 |
| GPPM | top | −0.14; 2,450 | −0.14; 2,750 | −0.08; 7,150 |

- With sorted or equal weights the growing pyramid's alignment exponents stay within −0.03 to
  +0.06, and random seeding stays at 13,800 everywhere.
- Tree levels with α give little (bottom-seeded exponents ≤ 0.05): their added levels hold
  almost no one, so they carry almost no influence.
- Mechanism check: under α = 2 each level's members are about α times as influential as the
  level below's, and the level is 0.8 times as large, so the bottom level's share of influence
  falls by 1.6x per added level. Three added levels over N = 300–10,000 predicts an exponent of
  ln(1.6³) / ln(33.3) = 0.40 for bottom seeding at q ≈ 0; measured 0.39. Incoherence breaks
  the stacking (0.10 at q = 1).
- The depth schedule was matched to the tree, not derived from a span-of-control rule; the
  exponent scales with how many levels are added per factor of N.

## Larger γ (`gamma_sweep.csv`, `ablation/gamma_check.py`)

γ = 0.01, 0.1, 0.3, 0.6, 0.9; new structure and old structure; all weightings and seedings;
q = 0.05 and 1.0; 8/8/6/4 graphs at N = 300/1,000/3,000/10,000.

- New structure: no size effect at any γ (every exponent within ±0.05).
- Old structure: random seeding picks one up from γ = 0.3 (q = 1: 0.11 → 0.25 → 0.33 under
  α = 2 at γ = 0.3/0.6/0.9; 0.06 → 0.27 sorted). Strong coupling makes alignment depend on
  reaching the poorly connected parts, the same feature behind the old internal-consensus
  scaling.
- q matters more as γ grows, even with sorted or equal weights. New structure, top seeded,
  hierarchy ÷ non-hierarchy at N = 10,000: 1.04 / 1.25 / 1.6 / 1.85 / 1.96 (sorted) across the
  five γ. Under α the hierarchy's top-down advantage shrinks (0.51 → 0.90) and its bottom-up
  penalty grows (4.6 → 6.1).

## Density

Thinning GPPM edges to rule B's count changes little for alignment (top-seeded, equal weights:
27,900 against 30,100 at full density, against 15,600 for 6-subordinate edges), so the edge
effect on alignment comes from how edges are spread (top nodes get few), not from their
number. For internal consensus, thinned GPPM edges at q = 1 show some size effect
(0.02–0.12 depending on weights), probably from poorly connected nodes.
