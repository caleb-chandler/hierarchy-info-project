"""
Alignment figures for results/2026-11-04_alignment_T_vs_rewiring/, written to
figures/2026-11-04_alignment_T_vs_rewiring/ with a CSV of the plotted numbers in tables/.

  fig3_alignment_vs_N_alpha3         alignment time vs N at alpha = 3; rows top / bottom seeded (at
                                     alpha = 1 every structure is flat in N; the table has every alpha)
  fig4_alignment_exponent_vs_alpha   scaling exponent vs alpha; rows top / bottom seeded, with the
                                     alpha * r prediction for the pure hierarchy (bottom seeded)

Style and helpers: fig_style.py (same colors and labels as figures 1-2).
"""
import os

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator

from fig_style import (PROBE, AXIS, INK2, PANELS, ramp, fmt_param, label, slope_ci, figure, finish,
                       plot_line, end_labels, legend, n_axis, log_y)

RES = os.path.join(PROBE, 'results', '2026-11-04_alignment_T_vs_rewiring')
FIG = os.path.join(PROBE, 'figures', '2026-11-04_alignment_T_vs_rewiring')
TAB = os.path.join(FIG, 'tables')
os.makedirs(TAB, exist_ok=True)
R = 0.65
ROWS = [('top', 'Top 10% seeded'), ('bottom', 'Bottom 10% seeded')]

df = pd.read_csv(os.path.join(RES, 'alignment.csv'))
NS = sorted(df.N.unique())
DEPTH = df.groupby('N').depth.first()
q10k = df[(df.N == 10_000) & (df.alpha == 1) & (df.seed_rule == 'top')].groupby(['family', 'param']).q.median()
compact = lambda v: f"{v / 1e3:g}K" if v >= 1e3 else f"{v:g}"   # noqa: E731

# ===================================================================== figure 3
A3 = 3.0
med = df.groupby(['seed_rule', 'family', 'param', 'alpha', 'N']).t_align.median().rename('median_time').reset_index()
a1 = med[(med.alpha == 1) & (med.seed_rule != 'random')].groupby('seed_rule').median_time.agg(['min', 'max'])
fig, axes = figure(2, 5.2, 4.2,
                   "Continuous-time alignment time (steps until every node is within δ = 10⁻⁶ of the "
                   f"environment's opinion) at α = {A3:g}, median of 2–8 graphs per point. Sources: the top or "
                   "bottom 10% of nodes by trophic level, each giving γ = 0.01 of its attention to the environment. "
                   "Same networks as figures 1–2 (depth 5 → 19 levels, top axis). At α = 1 no structure's alignment "
                   f"time changes with N (top seeded {a1.loc['top', 'min']:,.0f}–{a1.loc['top', 'max']:,.0f} steps, "
                   f"bottom seeded {a1.loc['bottom', 'min']:,.0f}–{a1.loc['bottom', 'max']:,.0f}; every α is in the "
                   "table). Dotted: the floor ln(10⁶)/γ = 1,382 steps. The legend in each upper panel applies to "
                   "the panel below it.", nrows=2, sharey='row')
log_y(axes[0, 0], 1e3, 1e5, compact)
log_y(axes[1, 0], 1e4, 1e11)
axes[1, 0].yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{int(round(np.log10(v)))}}}$"))
for i, (rule, rule_title) in enumerate(ROWS):
    axes[i, 0].set_ylabel('Alignment time (steps)')
    for ax, (fam, title) in zip(axes[i], PANELS):
        d = med[(med.family == fam) & (med.alpha == A3) & (med.seed_rule == rule)]
        pars = sorted(d.param.unique())
        ends = []
        for v, c in zip(pars, ramp(len(pars))):
            m = d[d.param == v].set_index('N').median_time
            plot_line(ax, m.index, m.values, c, label(fam, v, q10k[(fam, v)]))
            ends.append((m.index[-1], m.values[-1], fmt_param(fam, v)))
        if rule == 'top':
            ax.axhline(np.log(1e6) / 0.01, color=INK2, linewidth=1.0, linestyle=(0, (1, 2)), zorder=1)
        ax.set_title(f"{rule_title} · {title[0].lower() + title[1:]}", pad=22)
        n_axis(ax, NS, DEPTH)
        if i == 0:
            legend(ax, 'upper left')
        end_labels(ax, ends)
finish(fig, f'Alignment time vs group size, α = {A3:g}', os.path.join(FIG, 'fig3_alignment_vs_N_alpha3.png'))
med.merge(q10k.rename('q_at_10k').reset_index(), on=['family', 'param']).assign(depth=lambda x: x.N.map(DEPTH)) \
   .to_csv(os.path.join(TAB, 'fig3_alignment_vs_N_all_alpha.csv'), index=False)

# ===================================================================== figure 4
rows = []
alphas = sorted(df.alpha.unique())
for (rule, fam, v, a), g in df[df.seed_rule.isin(['top', 'bottom'])].groupby(['seed_rule', 'family', 'param', 'alpha']):
    b, lo, hi = slope_ci(g, 't_align')
    rows.append(dict(seed_rule=rule, family=fam, param=v, q_at_10k=q10k[(fam, v)], alpha=a, exponent=b,
                     ci_low=lo, ci_high=hi, share_estimated=(g.method == 'estimate').mean(), n_graphs=len(g)))
ex = pd.DataFrame(rows)
ex['hierarchy_prediction'] = np.maximum(np.log(ex.alpha * R), 0) / np.log(1 / R)
half = (ex.ci_high - ex.ci_low) / 2
fig, axes = figure(2, 5.2, 4.2,
                   "Effective scaling exponent of alignment time: slope of log time on log N over every graph, "
                   "N = 300–100,000. Top-seeded exponents are negative because time falls toward its floor, "
                   "ln(10⁶)/γ = 1,382 steps, reached once the top 10% hold nearly all influence. Dashed: prediction "
                   "for the pure hierarchy, ln(α·r)/ln(1/r) when α·r > 1 (r = 0.65). Bottom-seeded hierarchy times "
                   "reach ~10¹⁴ steps at α = 6 and N = 100,000; once 1 − ρ < 10⁻⁹ they come from the weak-coupling "
                   f"estimate (0.4% of all rows). 95% intervals ±{half.min():.2f}–{half.max():.2f} (median "
                   f"±{half.median():.2f}, in the table); the widest are lightly rewired networks, whose few "
                   "shortcuts vary from graph to graph. The legend in each lower panel applies to the panel above it.",
                   nrows=2, sharey='row')
axes[0, 0].set_ylim(-0.30, 0.06)
axes[1, 0].set_ylim(-0.15, 3.7)
for i, (rule, rule_title) in enumerate(ROWS):
    axes[i, 0].set_ylabel('Scaling exponent (time ~ N^exponent)')
    for ax, (fam, title) in zip(axes[i], PANELS):
        e = ex[(ex.seed_rule == rule) & (ex.family == fam)]
        pars = sorted(e.param.unique())
        ends = []
        for v, c in zip(pars, ramp(len(pars))):
            h = e[e.param == v].sort_values('alpha')
            plot_line(ax, h.alpha, h.exponent, c, label(fam, v, q10k[(fam, v)]))
            ends.append((h.alpha.iloc[-1], h.exponent.iloc[-1], fmt_param(fam, v)))
        if rule == 'bottom':
            a = np.array(alphas)
            ax.plot(a, np.maximum(np.log(a * R), 0) / np.log(1 / R), color=INK2, linewidth=1.0,
                    linestyle=(0, (4, 3)), label='prediction for the hierarchy', zorder=1)
            legend(ax, 'upper left')
        ax.set_title(f"{rule_title} · {title[0].lower() + title[1:]}")
        ax.xaxis.set_major_locator(FixedLocator(alphas))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel('Deference α')
        ax.axhline(0, color=AXIS, linewidth=0.8, zorder=0)
        end_labels(ax, ends)
finish(fig, 'Alignment scaling exponent vs deference α', os.path.join(FIG, 'fig4_alignment_exponent_vs_alpha.png'))
ex.to_csv(os.path.join(TAB, 'fig4_alignment_exponent_vs_alpha.csv'), index=False)
