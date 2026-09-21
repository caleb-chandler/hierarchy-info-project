# Methods: equations and their source lines

Every equation below is tagged with the line(s) of code it was transcribed from.
The tags are the point of this document: verify each equation against its lines
rather than against the surrounding prose.

## Notation

| Symbol | Meaning |
|---|---|
| $G=(V,E)$ | directed graph; $(i,j)\in E$ means an edge $i \to j$, read "$i$ has authority over $j$" |
| $E_{\text{tree}}$ | the $N-B$ edges added during the tree phase (`tree_edge_set`) |
| $\ell_i$ | level of node $i$; basal nodes have $\ell_i = 1$ (`G.nodes[i]['level']`) |
| $\mathcal{B}, F$ | basal set $\{i : \ell_i = 1\}$ and its complement (the "free" nodes) |
| $S_\ell$ | non-basal nodes at level $\ell$ (`sources_by_level`) |
| $V_\ell$ | all nodes at level $\ell$ (`targets_by_level`) |
| $k_i^{\text{out}}$ | out-degree of $i$ (`out_degree`) |
| $[\,P\,]$ | Iverson bracket: $1$ if $P$ holds, else $0$ |
| $\alpha$ | influence multiplier on asymmetric ties (`ALPHA`) |
| $\gamma$ | fitted scaling exponent (called `alpha` in `figures.ipynb` — distinct from $\alpha$) |
| $T$ | kernel bandwidth (distinct from $T_c$, the consensus time) |

Two naming collisions exist in the codebase and are disambiguated here:
`alpha` means the influence multiplier in `simulation.py` but the scaling
exponent in `figures.ipynb`; and `T` (bandwidth) is plotted against a quantity
this document calls $T_c$ (consensus time).

## 1. Control parameters

`simulation.py:8-15`

$$N \in \{100,\dots,10^4\}\ \text{(20 log-spaced)}, \quad T \in \{0.1,0.2,\dots,1.5\}, \quad b=0.1, \quad \alpha=2, \quad M=20$$

`simulation.py:19-21` — the $N$ grid, deduplicated after rounding:

$$\mathcal{N} = \mathrm{unique}\big(\mathrm{round}(\mathrm{logspace}(\log_{10}100,\ \log_{10}10^4,\ 20))\big)$$

`generator.py:70-71` — node and edge counts per graph:

$$B = \max(1,\ \mathrm{round}(bN)), \qquad L = \mathrm{round}(cN)$$

`generator.py:28-33` — density calibrated once at $N_{\max}$, held fixed for
every $N$ in the sweep. With $B_{\max}=\mathrm{round}(bN_{\max})$ and margin $\mu=9$:

$$c = \underbrace{(1-b)}_{\text{tree edges}/N} + \underbrace{\frac{\mu B_{\max}\ln B_{\max}}{N_{\max}}}_{\text{coupon-collector excess}}$$

`generator.py:72-76` — feasibility constraint, enforced by raise:

$$L \ge N-B$$

## 2. Graph generation (GPPM)

### 2.1 Tree phase

`generator.py:80` — basal nodes $0,\dots,B-1$ with $\ell_i = 1$.

`generator.py:86-93` — for $n = B, B+1, \dots, N-1$ in increasing order, attach
uniformly to any node already placed:

$$j_n \sim \mathrm{Unif}\{0,1,\dots,n-1\}, \qquad E \leftarrow E \cup \{(n,j_n)\}, \qquad \ell_n = \ell_{j_n}+1$$

(The candidate pool is exactly $\{0,\dots,n-1\}$ because `available_nodes`
starts as the $B$ basal nodes and appends each $n$ after it is placed.)

Consequences used later: every edge originates at a non-basal node, and every
non-basal node has $k^{\text{out}} \ge 1$.

### 2.2 Excess-edge phase

Candidate set, `generator.py:112-127`:

$$\mathcal{C} = \{(i,j) : i \notin \mathcal{B},\ i \ne j,\ (i,j)\notin E_{\text{tree}}\}$$

Gaussian kernel on the level gap, peaked at a gap of $1$, `generator.py:132`:

$$w(\ell,\ell') = \exp\!\left(-\frac{(\ell-\ell'-1)^2}{2T^2}\right)$$

$$\Pr[(i,j)] \propto w(\ell_i,\ell_j), \qquad (i,j)\in\mathcal{C}$$

Because $w$ depends only on the level pair, sampling is bucketed by
$(\ell,\ell')$ with multiplicity `generator.py:123-127`:

$$n(\ell,\ell') = |S_\ell|\,|V_{\ell'}| - |V_{\ell'}|\,[\ell'=\ell] - |S_\ell|\,[\ell'=\ell-1]$$

then `generator.py:137-142`:

$$\Pr[\text{bucket }(\ell,\ell')] = \frac{w(\ell,\ell')\,n(\ell,\ell')}{\sum_{\ell,\ell'} w(\ell,\ell')\,n(\ell,\ell')}$$

with a uniform draw within the chosen bucket (`generator.py:162-163`). The
number of excess edges is `generator.py:149`:

$$L_{\text{excess}} = L - (N-B)$$

$T$ is the single structural knob: $T\to 0$ forces every excess edge to span
exactly one level; large $T$ flattens the kernel toward uniform level gaps.

### 2.3 Cleanup

`generator.py:177-179` — asymmetry is tagged after all edges exist:

$$\mathrm{asym}(i,j) = [\,(j,i)\notin E\,], \qquad (i,j)\in E$$

`generator.py:200-205` — restrict to the largest connected component of the
undirected projection, so $N_{\text{actual}} \le N$.

## 3. Influence matrix

`engine.py:39-51` — weight that $i$ places on $j$. Self-weight $1$; a
reciprocated tie is weight $1$ in both directions; on an asymmetric tie the
junior weights the senior at $\alpha$ and the senior weights the junior at $1$:

$$M_{ij} = [\,i=j\,] + [\,(i,j)\in E\,] + \alpha\,[\,(j,i)\in E\ \text{and}\ (i,j)\notin E\,]$$

`engine.py:53-56` — row-normalize to stochastic:

$$W = D^{-1}M, \qquad D = \mathrm{diag}\Big(\sum_j M_{ij}\Big)$$

Influence is bidirectional on every tie; edge direction only marks which side
is amplified.

## 4. Outcome: consensus time

`engine.py:176-198` — DeGroot dynamics (defined but **not** called in the
production pipeline; see §7):

$$x(t+1) = Wx(t), \qquad x(0)\sim\mathrm{Unif}(0,1)^N, \qquad \text{stop when } \max_i x_i - \min_i x_i < \delta$$

`engine.py:268-274` — the measured outcome, with $\lambda_2$ the
second-largest-modulus eigenvalue of $W$ (and $\lambda_1=1$), $\delta=10^{-6}$:

$$g = 1-|\lambda_2|, \qquad T_c = \frac{\ln\delta}{\ln|\lambda_2|} = \frac{\ln(1/\delta)}{\ln(1/|\lambda_2|)} \approx \frac{\ln(1/\delta)}{g}$$

$$T_c = \infty \quad\text{if}\quad |\lambda_2|\ge 1$$

## 5. Mediator: trophic incoherence

`engine.py:98-99` — basal membership is read from the level attribute, not from
out-degree:

$$\mathcal{B} = \{i : \ell_i = 1\}$$

`engine.py:111, 116` — prey-averaged levels, where "prey of $i$" means $i$'s
**out**-neighbors (the authority targets), reversing the usual ecological
convention:

$$s_i = 1 \quad (i \in \mathcal{B}), \qquad s_i = 1 + \frac{1}{k_i^{\text{out}}}\sum_{j:(i,j)\in E} s_j \quad (i \in F)$$

`engine.py:113-117` — as a linear system on the free block, with
$r_i = \#\{j\in\mathcal{B} : (i,j)\in E\}$:

$$\big(\mathrm{diag}(k_F^{\text{out}}) - A_{FF}\big)\,s_F = k_F^{\text{out}} + r_F$$

`engine.py:129-134` — per-edge level gap, its mean, and the incoherence
parameter ($\mathrm{std}$ is the population standard deviation, `ddof=0`):

$$x_{ij} = s_i - s_j, \qquad \bar x = 1 \ \text{(exactly)}, \qquad q = \mathrm{std}(x) = \sqrt{\overline{x^2}-1}$$

$\bar x = 1$ is exact rather than approximate because every edge originates at a
non-basal node (§2.1), so

$$\sum_{(i,j)\in E}(s_i-s_j) = \sum_{i\in F} k_i^{\text{out}}\Big(s_i - \tfrac{1}{k_i^{\text{out}}}\textstyle\sum_j s_j\Big) = \sum_{i\in F} k_i^{\text{out}} = |E|$$

This is checkable at runtime: `mean_trophic_distance` should print as $1.0$.

## 6. Estimation

`figures.ipynb` cell 1 — per cell $(N,T)$ over $M=20$ graph draws indexed $m$:

$$\widetilde{T_c}(N,T) = \operatorname{median}_m T_c^{(m)}, \qquad \text{band} = [Q_{25},\ Q_{75}]$$

Power law in system size, one exponent per $T$, fit by OLS in logs
(`np.polyfit(np.log(sizes), np.log(medians), 1)`):

$$\log \widetilde{T_c}(N,T) = \log C(T) + \gamma(T)\log N \iff \widetilde{T_c} \propto N^{\gamma(T)}$$

Incoherence pooled over all sizes and trials at a given $T$ (`all_q`):

$$\bar q(T) = \frac{1}{|\mathcal{N}|M}\sum_{N\in\mathcal{N}}\sum_{m=1}^{M} q^{(m)}(N,T)$$

Reported relations: $\gamma$ vs. $\bar q$ (cell 3), $\bar q$ vs. $T$ (cell 4),
$\bar q$ vs. $N$ (cell 5), $\widetilde{T_c}$ vs. $T/\bar q$ (cell 6),
$\widetilde{T_c}$ vs. $\bar q$ at fixed $N$ (cell 7).

## 7. Caveats that affect interpretation

1. **$T_c$ is spectral, not simulated.** `run_trial` (`engine.py:285-315`)
   calls `compute_spectral_gap` and `trophic_coherence` only. `simulate_degroot`
   is never invoked by `simulation.py`, so every reported consensus time is the
   asymptotic prediction $\ln\delta/\ln|\lambda_2|$, not an iterate count.

2. **$T \mapsto \bar q$ is not injective.** The fold-back is visible in
   `figures.ipynb` cell 4 and noted in cell 7. Two different $T$ values can give
   the same $\bar q$ from different graph ensembles, so sweeping $T$ alone does
   not identify $q$ as the cause of slow consensus. If $\gamma(\bar q)$ is
   single-valued across the fold, that is evidence $q$ is a sufficient
   statistic; if not, $T$ carries structure beyond $q$.

3. **$\bar q$ is pooled across $N$.** Cell 1 averages $q$ over the whole
   $N$-ensemble before plotting, while $q$ itself varies with $N$ (cell 5).
   The $x$-axis of the $\gamma$-vs-$\bar q$ plot is therefore an ensemble
   average, not a per-graph value.

4. **Excess-edge sampling is successive, not exact weighted-without-replacement.**
   `generator.py:152-173` draws buckets with replacement and rejects duplicates.
   The realized joint distribution over edge sets is the successive-sampling
   distribution, which differs from exact weighted WOR at second order. With
   $L_{\text{excess}} \ll |\mathcal{C}|$ the difference is small, but it is not zero.

5. **One RNG stream, consumed sequentially.** `simulation.py:36` creates a
   single `default_rng(21)` shared across all $(T,N,m)$ cells in loop order.
   Cells are reproducible only by rerunning the whole sweep in the same order,
   not independently.

6. **No confidence intervals are computed.** `np.polyfit` returns point
   estimates only; the exponents $\gamma(T)$ in cell 3 are plotted without
   uncertainty.

7. **$T=0.1$ is degenerate.** $\bar q \approx 8.5\times10^{-11}$, so $T/\bar q$
   diverges; cell 6 excludes it via a $\bar q > 10^{-6}$ filter.
