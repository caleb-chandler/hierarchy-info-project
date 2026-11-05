# Internal consensus at α = 10, 2026-11-05

## Setup

- **Graphs:** 217 of the 651 graphs of `results/2026-11-04_internal_T_vs_rewiring/`, rebuilt from the
  same seeds (pairing checked: identical q and node count for all 217).
  - T = 0.1 (hierarchy), 1, 1.5, 2, ∞ (flat); p = 0.03, 0.1.
  - Every N (300–100,000, 5–19 levels) with the 11-04 replicate counts.
- **Outcome:** continuous-time internal consensus at α = 10, next to α = 6 from the 11-04 run on the
  same graphs.
- **Measures:**
  - **Exponent:** OLS slope of log time on log N, ± 95% CI half-width.
  - **Cost per level:** OLS slope of time on depth.
  - **Ratio to flat:** median time at N = 100,000 over the flat network's at the same α.

Code: `T_vs_rewiring/run_alpha10.py` (115 s on 8 workers), `T_vs_rewiring/summarize_alpha10.py`.
Raw rows: `scaling_alpha10.csv`. Tables: `summary_tables.md`.

## Findings

1. **At α = 10 the coherent hierarchy beats the flat network.**

   | | exponent α = 6 | α = 10 | cost per level α = 6 | α = 10 | vs flat at 100k, α = 10 |
   |---|---|---|---|---|---|
   | hierarchy (T = 0.1) | 0.103 ± 0.016 | **0.018 ± 0.013** | 2.71 | **0.40** | **0.91** |
   | flat (T = ∞) | 0.056 ± 0.022 | 0.061 ± 0.023 | 1.10 | 1.19 | 1 |

   - The 95% intervals of the two exponents don't overlap.
   - The crossover falls between α = 6 and 10. The 11-04 statement that the hierarchy beats
     rank-ignoring networks "at no α" holds only up to α = 6.
   - The 0.91 ratio at N = 100,000 rests on 2 graphs. The exponent difference is the robust part.

2. **Local disorder stays much worse than either.**
   - T = 1.5 is 3.6x slower than flat at N = 100,000, with a cost of 11.5 per level.
   - T = 2 is unchanged from α = 6 (11.7 per level). T = 1 halves its cost to 5.7.
   - So the worst-T peak is moving to larger T as α grows.

3. **Light rewiring is now worse than the hierarchy too.** p = 0.03 is 2.05x slower than flat and
   p = 0.1 is 1.31x slower. So at α = 10, the coherent hierarchy is the fastest structure tested.

## Not yet checked

These times come from the slowest mode alone. At steep α the network acts like a chain of followers,
and the delay as an opinion passes down 19 levels could add a depth cost the slowest mode misses. The
check is to run the dynamics on a few hierarchy graphs and compare.
