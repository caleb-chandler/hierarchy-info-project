"""
Internal-consensus figures for results/2026-11-04_internal_T_vs_rewiring/, written to
figures/2026-11-04_internal_T_vs_rewiring/ with a CSV of the plotted numbers in tables/.

  fig1_time_vs_N_alpha{A}    internal consensus time vs N at alpha = FIG1_ALPHA, one line per T / p value
  fig2_exponent_vs_alpha     scaling exponent vs alpha, one line per T / p value
  fig5_exponent_vs_disorder  the same exponents vs T / p, one line per alpha (fig3-4 are alignment)

Style and helpers: fig_style.py. Exponent: OLS slope of log(time) on log(N) over every graph
(N = 300-100,000), with a 95% interval of +-1.96 standard errors (in the CSV).
"""
import os

import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

from fig_style import (PROBE, AXIS, PANELS, ramp, alpha_ramp, fmt_param, label, slope_ci, figure,
                       finish, plot_line, plot_broken, end_labels, legend, n_axis, disorder_axis,
                       log_y)

RES = os.path.join(PROBE, 'results', '2026-11-04_internal_T_vs_rewiring')
FIG = os.path.join(PROBE, 'figures', '2026-11-04_internal_T_vs_rewiring')
TAB = os.path.join(FIG, 'tables')
os.makedirs(TAB, exist_ok=True)
# deference used in figure 1 (data filter, title and file names)
FIG1_ALPHA = 2

df = pd.read_csv(os.path.join(RES, 'scaling.csv'))
NS = sorted(df.N.unique())
DEPTH = df.groupby('N').depth.first()
q10k = df[(df.N == 10_000) & (df.alpha == 2)].groupby(
    ['family', 'param']).q.median()

# ===================================================================== figure 1
rows = []
fig, axes = figure(2, 5.2, 4.6, ' ')
axes = axes[0]
log_y(axes[0], 35, 1000)
axes[0].set_ylabel('Consensus time (steps)')
for ax, (fam, title) in zip(axes, PANELS):
    d = df[(df.family == fam) & (df.alpha == FIG1_ALPHA)]
    pars = sorted(d.param.unique())
    ends = []
    for v, c in zip(pars, ramp(len(pars))):
        m = d[d.param == v].groupby('N').t_cont.median()
        plot_line(ax, m.index, m.values, c, label(fam, v, q10k[(fam, v)]))
        ends.append((m.index[-1], m.values[-1], fmt_param(fam, v)))
        rows += [dict(family=fam, param=v, q_at_10k=q10k[(fam, v)], N=n, depth=DEPTH[n], median_time=t,
                      n_graphs=int((d[(d.param == v) & (d.N == n)]).shape[0]))
                 for n, t in m.items()]
    # ax.set_title(title, pad=22)
    n_axis(ax, NS, DEPTH)
    legend(ax, 'upper left')
    end_labels(ax, ends)
finish(fig, f'Consensus time vs group size, α = {FIG1_ALPHA}',
       os.path.join(FIG, f'fig1_time_vs_N_alpha{FIG1_ALPHA}.png'))
pd.DataFrame(rows).to_csv(os.path.join(
    TAB, f'fig1_time_vs_N_alpha{FIG1_ALPHA}.csv'), index=False)

# ===================================================================== figure 2
rows = []
alphas = sorted(df.alpha.unique())
for (fam, v, a), g in df.groupby(['family', 'param', 'alpha']):
    b, lo, hi = slope_ci(g, 't_cont')
    rows.append(dict(family=fam, param=v, q_at_10k=q10k[(fam, v)], alpha=a, exponent=b, ci_low=lo, ci_high=hi,
                     n_graphs=len(g)))
ex = pd.DataFrame(rows)
half = ((ex.ci_high - ex.ci_low) / 2)
fig, axes = figure(2, 5.2, 4.6, ' ')

axes = axes[0]
axes[0].set_ylim(-0.02, 0.80)
axes[0].set_ylabel('Scaling exponent')
for ax, (fam, title) in zip(axes, PANELS):
    e = ex[ex.family == fam]
    pars = sorted(e.param.unique())
    ends = []
    for v, c in zip(pars, ramp(len(pars))):
        h = e[e.param == v].sort_values('alpha')
        plot_line(ax, h.alpha, h.exponent, c, label(fam, v, q10k[(fam, v)]))
        ends.append((h.alpha.iloc[-1], h.exponent.iloc[-1], fmt_param(fam, v)))
#    ax.set_title(title)
    ax.xaxis.set_major_locator(FixedLocator(alphas))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel('α')
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=0)
    legend(ax, 'upper right')
    end_labels(ax, ends)
finish(fig, 'Scaling exponent vs deference α',
       os.path.join(FIG, 'fig2_exponent_vs_alpha.png'))
ex.to_csv(os.path.join(TAB, 'fig2_exponent_vs_alpha.csv'), index=False)

# ===================================================================== figure 5
XLABEL = {'T': 'Kernel width T  (hierarchy → egalitarianism)',
          'p': 'Rewiring fraction p  (hierarchy → egalitarianism)'}
fig, axes = figure(2, 5.2, 4.6, ' ')
axes = axes[0]
axes[0].set_ylim(-0.02, 0.80)
axes[0].set_ylabel('Scaling exponent')
for ax, (fam, title) in zip(axes, PANELS):
    e = ex[ex.family == fam]
    for a, c in zip(alphas, alpha_ramp(len(alphas))):
        h = e[e.alpha == a].sort_values('param')
        plot_broken(ax, fam, h.param, h.exponent, c, f"α = {a:g}")
#    ax.set_title(title)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=0)
    disorder_axis(ax, fam, e.param.unique(), XLABEL[fam])
legend(axes[0], 'upper right')
finish(fig, 'Scaling exponent vs hierarchy',
       os.path.join(FIG, 'fig5_exponent_vs_hierarchy.png'))
ex.sort_values(['family', 'alpha', 'param']).to_csv(
    os.path.join(TAB, 'fig5_exponent_vs_hierarchy.csv'), index=False)
