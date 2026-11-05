"""
Figures and summary tables for the 2026-10-29 overnight run.

Usage: python make_figures.py RESULTS_DIR FIGURES_DIR

Every figure has a CSV twin in FIGURES_DIR/tables with the plotted numbers.
Exponents are OLS slopes of log(time) on log(N) over every graph in the cell
(all sizes, all repeats), with a 95% interval of +-1.96 standard errors.
"""
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator, FixedLocator   # noqa: E402

RES, FIG = sys.argv[1], sys.argv[2]
TAB = os.path.join(FIG, 'tables')
os.makedirs(TAB, exist_ok=True)

# --- palette (reference instance, validated: categorical slots 1-3 all-pairs; ordinal blue ramps) ---
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
RULES = ['top', 'random', 'bottom']
RULE_COLOR = {'top': '#2a78d6', 'bottom': '#eb6834', 'random': '#1baf7a'}
RULE_LABEL = {'top': 'Top 10% seeded', 'random': 'Random 10% seeded', 'bottom': 'Bottom 10% seeded'}
SIGMA_COLOR = {0.0: '#86b6ef', 0.5: '#5598e7', 1.0: '#2a78d6', 1.5: '#1c5cab', 2.0: '#104281'}
RAMP4 = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
RAMP3 = ['#6da7ec', '#256abf', '#0d366b']

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9.5,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 10.5, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': MUTED, 'ytick.labelcolor': MUTED,
    'lines.linewidth': 1.6, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'legend.frameon': False, 'legend.fontsize': 8.5, 'text.color': INK,
})


# --- helpers -----------------------------------------------------------------

def compact(v, _=None):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}K"
    return f"{v:g}"


def logaxis(ax, which):
    axis = ax.xaxis if which == 'x' else ax.yaxis
    (ax.set_xscale if which == 'x' else ax.set_yscale)('log')
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(compact))
    axis.set_minor_formatter(NullFormatter())


def q_axis(ax, qs, label=True):
    ax.set_xscale('log')
    ticks = [q for q in [0.05, 0.1, 0.2, 0.5, 1.0] if min(qs) * 0.95 <= q <= max(qs) * 1.05] + [max(qs)]
    ax.xaxis.set_major_locator(FixedLocator(sorted(set(np.round(ticks, 3)))))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if label:
        ax.set_xlabel('Trophic incoherence q (target)')


def line(ax, x, med, lo=None, hi=None, color=INK2, label=None):
    if lo is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.plot(x, med, color=color, label=label, marker='o', markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2)


def slope_ci(df, y='align_time'):
    lx, ly = np.log(df['N'].to_numpy(float)), np.log(df[y].to_numpy(float))
    if len(lx) < 3 or np.ptp(lx) == 0:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(lx, ly, 1)
    se = np.sqrt(np.var(ly - (a + b * lx), ddof=2) / ((lx - lx.mean()) ** 2).sum())
    return b, b - 1.96 * se, b + 1.96 * se


def quantiles(g, col):
    return g[col].median(), g[col].quantile(0.25), g[col].quantile(0.75)


def figure(ncols, nrows=1, w=4.3, h=3.4, sharey=True, subtitle='', legend=False):
    '''Panels plus a header band (in inches) sized for the title, the wrapped subtitle and a legend row.'''
    W = w * ncols
    lines = textwrap.wrap(subtitle, width=int(W * 15))
    legend_top = 0.50 + 0.19 * len(lines) + 0.08
    head = legend_top + (0.34 if legend else 0) + 0.36
    H = h * nrows + head + 0.58
    fig, axes = plt.subplots(nrows, ncols, figsize=(W, H), sharey=sharey, squeeze=False)
    fig.subplots_adjust(top=1 - head / H, bottom=0.58 / H, left=0.9 / W, right=1 - 0.15 / W,
                        wspace=0.12 if sharey is True else 0.22, hspace=0.42 * 3.4 / h)
    fig._header = dict(lines=lines, legend_top=legend_top)
    return fig, axes


def finish(fig, title, name):
    H = fig.get_size_inches()[1]
    hdr = fig._header
    fig.text(0.012, 1 - 0.16 / H, title, fontsize=12.5, weight='semibold', color=INK, va='top')
    fig.text(0.012, 1 - 0.50 / H, '\n'.join(hdr['lines']), fontsize=9, color=INK2, va='top', linespacing=1.4)
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=200)
    plt.close(fig)
    print('wrote', name)


def fig_legend(fig, handles):
    H = fig.get_size_inches()[1]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.004, 1 - fig._header['legend_top'] / H),
               ncol=len(handles), handlelength=1.8, columnspacing=1.6)


def key(color, label):
    # legend key without the marker ring, so the line doesn't read as dashed
    return Line2D([], [], color=color, marker='o', markersize=5, markeredgewidth=0, label=label)


def panel_legend(ax):
    handles = [key(h.get_color(), h.get_label()) for h in ax.get_lines() if not h.get_label().startswith('_')]
    ax.legend(handles=handles, loc='best')


def load(name):
    path = os.path.join(RES, name + '.csv')
    return pd.read_csv(path) if os.path.exists(path) else None


def save_table(rows, name):
    pd.DataFrame(rows).to_csv(os.path.join(TAB, name + '.csv'), index=False)


# ===================================================================== test 1
d1 = load('test1_hierarchy_q0')
if d1 is not None:
    rows = []
    fig, axes = figure(2, subtitle="γ = 0.01 share of each source's attention; 8 levels, r = 0.8, c = 6. Median and "
                                   "interquartile range of 20 graphs per size; exponent = slope of log time on log N, "
                                   "with its 95% interval.")
    for ax, (sigma, title) in zip(axes[0], [(0.5, 'Sorted lognormal weights, σ = 0.5 (baseline)'),
                                            (0.0, 'Equal weights (σ = 0), for reference')]):
        for rule in RULES:
            sub = d1[(d1.sigma == sigma) & (d1.seed_rule == rule)]
            med, lo, hi = quantiles(sub.groupby('N'), 'align_time')
            b, bl, bh = slope_ci(sub)
            line(ax, med.index, med, lo, hi, RULE_COLOR[rule],
                 f"{RULE_LABEL[rule]}: exponent {b:+.2f} [{bl:+.2f}, {bh:+.2f}]")
            rows += [dict(sigma=sigma, seed_rule=rule, N=n, median=m, q25=l, q75=h, exponent=b, ci_lo=bl, ci_hi=bh)
                     for n, m, l, h in zip(med.index, med, lo, hi)]
        logaxis(ax, 'x')
        logaxis(ax, 'y')
        ax.set_title(title)
        ax.set_xlabel('Group size N')
        panel_legend(ax)
    axes[0][0].set_ylabel('Time to align with the environment')
    finish(fig, 'Test 1 · Aligning a pure hierarchy (q = 0) with outside information', 't1_alignment_vs_N')
    save_table(rows, 't1_alignment_vs_N')

    fig, axes = figure(1, w=6.2, subtitle='No environment input. Regime check: alignment reflects where the sources '
                                          'sit only while it is much slower than this.')
    ax = axes[0][0]
    rows = []
    for sigma, color, label in [(0.5, '#2a78d6', 'Sorted weights, σ = 0.5'), (0.0, '#eb6834', 'Equal weights')]:
        sub = d1[d1.sigma == sigma].drop_duplicates(['N', 'rep'])
        med, lo, hi = quantiles(sub.groupby('N'), 'internal_time')
        b, bl, bh = slope_ci(sub, 'internal_time')
        line(ax, med.index, med, lo, hi, color, f"{label}: exponent {b:+.2f} [{bl:+.2f}, {bh:+.2f}]")
        rows += [dict(sigma=sigma, N=n, median=m, q25=l, q75=h, exponent=b) for n, m, l, h in zip(med.index, med, lo, hi)]
    logaxis(ax, 'x')
    ax.set_xlabel('Group size N')
    ax.set_ylabel('Internal consensus time')
    panel_legend(ax)
    finish(fig, 'Test 1 · Internal consensus at q = 0', 't1_internal_vs_N')
    save_table(rows, 't1_internal_vs_N')

# ===================================================================== test 2
d2 = load('test2_random_seeding')
if d2 is not None:
    qs = sorted(d2.q_target.unique())
    sizes = np.array(sorted(d2.N.unique()))
    Ns = sorted({int(sizes[np.argmin(np.abs(np.log(sizes / target)))]) for target in [100, 1000, 10000]})
    fig, axes = figure(3, subtitle='Sorted σ = 0.5 weights, γ = 0.01. Median and interquartile range of 10 graphs '
                                   'per cell, shown for three group sizes.', legend=True)
    rows = []
    for ax, rule in zip(axes[0], ['random', 'top', 'bottom']):
        for N, color in zip(Ns, RAMP3[-len(Ns):]):
            med, lo, hi = quantiles(d2[(d2.seed_rule == rule) & (d2.N == N)].groupby('q_target'), 'align_time')
            line(ax, med.index, med, lo, hi, color)
            rows += [dict(seed_rule=rule, N=N, q_target=q, median=m, q25=l, q75=h)
                     for q, m, l, h in zip(med.index, med, lo, hi)]
        q_axis(ax, qs)
        logaxis(ax, 'y')
        ax.set_title(RULE_LABEL[rule])
    axes[0][0].set_ylabel('Time to align with the environment')
    fig_legend(fig, [key(c, f"N = {N:,}") for N, c in zip(Ns, RAMP3[-len(Ns):])])
    finish(fig, 'Test 2 · Alignment time across coherence', 't2_alignment_vs_q')
    save_table(rows, 't2_alignment_vs_q')

    fig, axes = figure(1, w=6.4, subtitle='Slope of log alignment time on log N over N = 100–10,000, 95% interval '
                                          'shaded; 0 means group size has no effect.')
    ax = axes[0][0]
    rows = []
    for rule in ['random', 'top', 'bottom']:
        ex = [(q, *slope_ci(d2[(d2.seed_rule == rule) & (d2.q_target == q)])) for q in qs]
        q_, b, bl, bh = map(np.array, zip(*ex))
        line(ax, q_, b, bl, bh, RULE_COLOR[rule], RULE_LABEL[rule])
        rows += [dict(seed_rule=rule, q_target=a, exponent=x, ci_lo=y, ci_hi=z,
                      exponent_N300_up=slope_ci(d2[(d2.seed_rule == rule) & (d2.q_target == a) & (d2.N >= 300)])[0])
                 for a, x, y, z in ex]
    ax.axhline(0, color=AXIS, linewidth=0.9, zorder=0)
    q_axis(ax, qs)
    ax.set_ylabel('Exponent of alignment time vs N')
    panel_legend(ax)
    finish(fig, 'Test 2 · How alignment time scales with group size', 't2_exponent_vs_q')
    save_table(rows, 't2_exponent_vs_q')

# ===================================================================== test 3
d3 = load('test3_weight_distribution')
if d3 is not None:
    qs = sorted(d3.q_target.unique())
    Nmax = d3.N.max()
    sigmas = sorted(d3.sigma.unique())
    panels = [('sorted', 'weights sorted by level'), ('random', 'same weights shuffled')]

    def cell(rule, placement, sigma):
        pl = 'sorted' if sigma == 0 else placement          # sigma = 0 is identical under both placements
        return d3[(d3.seed_rule == rule) & (d3.placement == pl) & (d3.sigma == sigma)]

    for what in ['exponent', 'time']:
        sub_txt = ('Slope of log alignment time on log N (N = 100–10,000, 10 graphs per cell, 95% interval shaded). '
                   'σ sets how unequal influence is; σ = 0 is equal weights and is the same in both columns.'
                   if what == 'exponent' else
                   f'Median and interquartile range of 10 graphs per cell at N = {Nmax:,}. σ = 0 is equal weights '
                   'and is the same in both columns.')
        fig, axes = figure(2, nrows=3, w=4.7, h=2.8, sharey='row', subtitle=sub_txt, legend=True)
        rows = []
        for r_i, rule in enumerate(['random', 'top', 'bottom']):
            for c_i, (placement, ptitle) in enumerate(panels):
                ax = axes[r_i][c_i]
                for sigma in sigmas:
                    sub = cell(rule, placement, sigma)
                    if what == 'exponent':
                        ex = [(q, *slope_ci(sub[sub.q_target == q])) for q in qs]
                        q_, b, bl, bh = map(np.array, zip(*ex))
                        line(ax, q_, b, bl, bh, SIGMA_COLOR[sigma])
                        rows += [dict(seed_rule=rule, placement=placement, sigma=sigma, q_target=a, exponent=x,
                                      ci_lo=y, ci_hi=z,
                                      exponent_N300_up=slope_ci(sub[(sub.q_target == a) & (sub.N >= 300)])[0])
                                 for a, x, y, z in ex]
                    else:
                        med, lo, hi = quantiles(sub[sub.N == Nmax].groupby('q_target'), 'align_time')
                        line(ax, med.index, med, lo, hi, SIGMA_COLOR[sigma])
                        rows += [dict(seed_rule=rule, placement=placement, sigma=sigma, N=Nmax, q_target=q,
                                      median=m, q25=l, q75=h) for q, m, l, h in zip(med.index, med, lo, hi)]
                q_axis(ax, qs, label=(r_i == 2))
                if what == 'exponent':
                    ax.axhline(0, color=AXIS, linewidth=0.9, zorder=0)
                else:
                    logaxis(ax, 'y')
                ax.set_title(f"{RULE_LABEL[rule]}, {ptitle}", fontsize=9.5)
            axes[r_i][0].set_ylabel('Exponent vs N' if what == 'exponent' else 'Time to align')
        fig_legend(fig, [key(SIGMA_COLOR[s], f"σ = {s:g}") for s in sigmas])
        if what == 'exponent':
            finish(fig, 'Test 3 · Does organizing influence along the hierarchy change size scaling?', 't3_exponent_vs_q')
        else:
            finish(fig, f'Test 3 · Alignment time at N = {Nmax:,}, by influence inequality and placement',
                   't3_alignment_vs_q')
        save_table(rows, 't3_exponent_vs_q' if what == 'exponent' else 't3_alignment_vs_q')

    fig, axes = figure(3, subtitle=f'Ratio of median alignment times at N = {Nmax:,}. Below 1: placing the largest '
                                   'weights at the top of the hierarchy makes alignment faster than the same weights '
                                   'shuffled.', legend=True)
    rows = []
    for ax, rule in zip(axes[0], ['random', 'top', 'bottom']):
        for sigma in [s for s in sigmas if s > 0]:
            a = cell(rule, 'sorted', sigma).query('N == @Nmax').groupby('q_target').align_time.median()
            b = cell(rule, 'random', sigma).query('N == @Nmax').groupby('q_target').align_time.median()
            ratio = (a / b).dropna()
            line(ax, ratio.index, ratio, color=SIGMA_COLOR[sigma])
            rows += [dict(seed_rule=rule, sigma=sigma, N=Nmax, q_target=q, sorted_over_shuffled=v)
                     for q, v in ratio.items()]
        ax.axhline(1, color=AXIS, linewidth=0.9, zorder=0)
        q_axis(ax, qs)
        logaxis(ax, 'y')
        ax.set_title(RULE_LABEL[rule])
    axes[0][0].set_ylabel('Sorted ÷ shuffled alignment time')
    fig_legend(fig, [key(SIGMA_COLOR[s], f"σ = {s:g}") for s in sigmas if s > 0])
    finish(fig, 'Test 3 · Effect of placing influence by rank', 't3_sorted_vs_shuffled')
    save_table(rows, 't3_sorted_vs_shuffled')

# ===================================================================== test 4
d4 = load('test4_depth')
if d4 is not None:
    qs_all = sorted(d4.q_target.unique())
    pick = [qs_all[i] for i in sorted({0, len(qs_all) * 4 // 9, len(qs_all) * 7 // 9, len(qs_all) - 1})]
    Nmax = d4.N.max()
    ramp = RAMP4[-len(pick):]
    Ds = sorted(d4.D.unique())

    def depth_axis(ax):
        ax.set_xticks(Ds)
        ax.set_xlabel('Number of levels')

    fig, axes = figure(3, subtitle=f'N = {Nmax:,}; level ratio r = 0.8 and c = 6 held fixed; sorted σ = 0.5 weights, '
                                   'γ = 0.01. Median and interquartile range of 5 graphs.', legend=True)
    rows = []
    for ax, rule in zip(axes[0], ['top', 'random', 'bottom']):
        for q, color in zip(pick, ramp):
            med, lo, hi = quantiles(d4[(d4.seed_rule == rule) & (d4.q_target == q) & (d4.N == Nmax)].groupby('D'),
                                    'align_time')
            line(ax, med.index, med, lo, hi, color)
            rows += [dict(seed_rule=rule, q_target=q, N=Nmax, D=D, median=m, q25=l, q75=h)
                     for D, m, l, h in zip(med.index, med, lo, hi)]
        depth_axis(ax)
        logaxis(ax, 'y')
        ax.set_title(RULE_LABEL[rule])
    axes[0][0].set_ylabel('Time to align with the environment')
    fig_legend(fig, [key(c, f"q = {q:.2g}") for q, c in zip(pick, ramp)])
    finish(fig, 'Test 4 · Does adding levels speed up alignment, in hierarchies vs non-hierarchies?',
           't4_alignment_vs_depth')
    save_table(rows, 't4_alignment_vs_depth')

    fig, axes = figure(3, subtitle='Slope of log alignment time on log N over N = '
                                   f'{", ".join(f"{n:,}" for n in sorted(d4.N.unique()))} (5 graphs each), '
                                   '95% interval shaded.', legend=True)
    rows = []
    for ax, rule in zip(axes[0], ['top', 'random', 'bottom']):
        for q, color in zip(pick, ramp):
            ex = [(D, *slope_ci(d4[(d4.seed_rule == rule) & (d4.q_target == q) & (d4.D == D)])) for D in Ds]
            D_, b, bl, bh = map(np.array, zip(*ex))
            line(ax, D_, b, bl, bh, color)
            rows += [dict(seed_rule=rule, q_target=q, D=a, exponent=x, ci_lo=y, ci_hi=z) for a, x, y, z in ex]
        ax.axhline(0, color=AXIS, linewidth=0.9, zorder=0)
        depth_axis(ax)
        ax.set_title(RULE_LABEL[rule])
    axes[0][0].set_ylabel('Exponent of alignment time vs N')
    fig_legend(fig, [key(c, f"q = {q:.2g}") for q, c in zip(pick, ramp)])
    finish(fig, 'Test 4 · Size scaling of alignment at each depth', 't4_exponent_vs_depth')
    save_table(rows, 't4_exponent_vs_depth')

    fig, axes = figure(1, w=6.2, subtitle=f'N = {Nmax:,}, no environment input; sorted σ = 0.5 weights. Median and '
                                          'interquartile range of 5 graphs.')
    ax = axes[0][0]
    rows = []
    sub = d4[d4.N == Nmax].drop_duplicates(['D', 'q_target', 'rep'])
    for q, color in zip(pick, ramp):
        med, lo, hi = quantiles(sub[sub.q_target == q].groupby('D'), 'internal_time')
        line(ax, med.index, med, lo, hi, color, f"q = {q:.2g}")
        rows += [dict(q_target=q, N=Nmax, D=D, median=m, q25=l, q75=h) for D, m, l, h in zip(med.index, med, lo, hi)]
    depth_axis(ax)
    ax.set_ylabel('Internal consensus time')
    panel_legend(ax)
    finish(fig, 'Test 4 · Internal consensus by depth', 't4_internal_vs_depth')
    save_table(rows, 't4_internal_vs_depth')

# ===================================================================== calibration check
checks = []
for name, df in [('test2', d2), ('test3', d3), ('test4', d4)]:
    if df is None:
        continue
    per = df.drop_duplicates(['D', 'N', 'q_index', 'rep']).groupby(['D', 'N', 'q_target']).q.median().reset_index()
    per['abs_dev'] = (per.q - per.q_target).abs()
    for (D, N), g in per.groupby(['D', 'N']):
        checks.append(dict(test=name, D=D, N=N, max_abs_dev=g.abs_dev.max(), median_abs_dev=g.abs_dev.median()))
if checks:
    save_table(checks, 'q_target_vs_measured')
    print('wrote q_target_vs_measured')
