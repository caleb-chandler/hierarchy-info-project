"""
Shared style and helpers for the T_vs_rewiring figures (make_figures.py, make_alignment_figures.py).

Style follows ruleB_overnight/make_figures.py. Parameter values are ordered, so each panel uses a
one-hue ordinal ramp (blue steps 250-700): darker = more hierarchical. The shared endpoints get the
same color in both families (T = 0.1 and p = 0 darkest, T = inf and p = 1 lightest). That range holds
about 8 steps far enough apart to tell by shade (OKLCH delta L >= 0.06), fewer than the 10-11 lines
per panel, so every line also carries its value as a label at its end.
"""
import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_hex   # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator, LogLocator   # noqa: E402
from matplotlib.transforms import offset_copy       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.dirname(HERE)

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95',
        '#104281', '#0d366b']                       # blue steps 250-700
plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans'], 'font.size': 9.5,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 10.5, 'axes.titleweight': 'semibold', 'axes.titlelocation': 'left',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': MUTED, 'ytick.labelcolor': MUTED,
    'lines.linewidth': 1.6, 'lines.solid_capstyle': 'round', 'lines.solid_joinstyle': 'round',
    'legend.frameon': False, 'legend.fontsize': 8, 'text.color': INK,
})
PANELS = [('T', 'Local disorder: kernel width T'),
          ('p', 'Shortcuts: rewiring fraction p')]


def ramp(n):
    '''n colors from the ordinal ramp, darkest first (most hierarchical first).'''
    cmap = plt.get_cmap('viridis_r')
    return [to_hex(cmap(x)) for x in np.linspace(1, 0, n)]


def alpha_ramp(n):
    '''n colors for the alpha values, lightest (alpha = 1) first. One-hue orange so they can't be
    read as the T / p colors of ramp(); Oranges 0.45-1.0 passes the ordinal palette checks.'''
    cmap = plt.get_cmap('Oranges')
    return [to_hex(cmap(x)) for x in np.linspace(0.45, 1.0, n)]


def fmt_param(fam, v):
    return '∞' if np.isinf(v) else f"{v:g}"


def label(fam, v, q):
    end = ' (hierarchy)' if (fam == 'T' and v == 0.1) or (fam == 'p' and v == 0) else \
          ' (egalitarianism)' if (fam == 'T' and np.isinf(v)) or (
              fam == 'p' and v == 1) else ''
    return f"{fam} = {fmt_param(fam, v)}{end}  ·  q {q:.2f}"


def slope_ci(g, col):
    '''OLS slope of log(col) on log(N) with a 95% interval of +-1.96 standard errors.'''
    lx, ly = np.log(g.N.to_numpy(float)), np.log(g[col].to_numpy(float))
    b, a = np.polyfit(lx, ly, 1)
    se = np.sqrt(np.var(ly - (a + b * lx), ddof=2) /
                 ((lx - lx.mean()) ** 2).sum())
    return b, b - 1.96 * se, b + 1.96 * se


def figure(ncols, w, h, subtitle, nrows=1, sharey=True, row_gap=1.0):
    '''Panels plus a header band (in inches) sized for the title and the wrapped subtitle.'''
    W = w * ncols
    lines = textwrap.wrap(subtitle, width=int(W * 15))
    head = 0.50 + 0.19 * len(lines) + 0.40
    H = h * nrows + row_gap * (nrows - 1) + head + 0.62
    fig, axes = plt.subplots(nrows, ncols, figsize=(
        W, H), sharey=sharey, squeeze=False)
    fig.subplots_adjust(top=1 - head / H, bottom=0.62 / H, left=0.8 / W, right=1 - 0.5 / W, wspace=0.16,
                        hspace=row_gap / h)
    fig._lines = lines
    return fig, axes


def finish(fig, title, path):
    H = fig.get_size_inches()[1]
    fig.text(0.012, 1 - 0.16 / H, title, fontsize=12.5,
             weight='semibold', color=INK, va='top')
    fig.text(0.012, 1 - 0.50 / H, '\n'.join(fig._lines),
             fontsize=9, color=INK2, va='top', linespacing=1.4)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print('wrote', os.path.basename(path))


def plot_line(ax, x, y, color, lab):
    ax.plot(x, y, color=color, label=lab, marker='o', markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0)


def end_labels(ax, items, gap_pt=8.5):
    '''Value labels just right of the plot, one per line, spread so they don't overlap, with a thin
    leader to each line's last point. items: (x_end, y_end, text). Axis limits must be final.'''
    gap = gap_pt * ax.figure.dpi / 72
    ys = np.array([ax.transData.transform((x, y))[1] for x, y, _ in items])
    order = np.argsort(ys)
    placed = ys[order].copy()
    # push apart, recentre, push apart again
    for _ in range(2):
        for i in range(1, len(placed)):
            placed[i] = max(placed[i], placed[i - 1] + gap)
        placed -= (placed - ys[order]).mean() / 2
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    inv = ax.transData.inverted()
    for k, idx in enumerate(order):
        x, y, text = items[idx]
        ax.annotate(text, xy=(x, y), xytext=(1.035, inv.transform((0, placed[k]))[1]),
                    textcoords=ax.get_yaxis_transform(), fontsize=7.5, color=INK2, va='center',
                    ha='left', annotation_clip=False,
                    arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.6, shrinkA=0, shrinkB=1.5))


def legend(ax, loc):
    handles = [Line2D([], [], color=h.get_color(), marker='o' if h.get_marker() == 'o' else None,
                      markersize=4.5, markeredgewidth=0, linestyle=h.get_linestyle(), label=h.get_label())
               for h in ax.get_lines() if not h.get_label().startswith('_')]
    ax.legend(handles=handles, loc=loc, handlelength=1.6, labelspacing=0.35)


def n_axis(ax, Ns, depth):
    '''Log N axis with the number of levels at each N along the top.'''
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(Ns))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v / 1e3:g}K" if v >= 1e3 else f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Group size N')
    top = ax.secondary_xaxis('top')
    top.set_xticks(Ns)
    top.set_xticklabels([str(depth[n]) for n in Ns])
    top.xaxis.set_minor_formatter(NullFormatter())
    top.tick_params(colors=AXIS, labelcolor=MUTED, labelsize=8)
    top.spines['top'].set_visible(False)
    ax.text(-0.012, 1.012, 'levels', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color=MUTED)


# T and p on a log x axis. Their off-scale ends (T = inf, p = 0) sit past an axis break.
BREAK = {'T': (np.inf, 16.0), 'p': (0.0, 0.001)}       # (value, where it is drawn)
TICKED = {'T': [0.1, 0.3, 0.5, 1, 2, 3, 5, 8, np.inf],  # labelled values; the rest get bare ticks
          'p': [0, 0.003, 0.01, 0.03, 0.1, 0.2, 0.5, 1]}


def xpos(fam, v):
    '''Plot positions of T or p values.'''
    end, at = BREAK[fam]
    v = np.asarray(v, float)
    return np.where(np.isclose(v, end), at, v)


def disorder_axis(ax, fam, values, xlabel):
    '''Log x axis with a tick at every T / p value, a readable subset labelled, and an axis break
    before the off-scale end.'''
    values = np.sort(np.asarray(values, float))
    ax.set_xscale('log')
    ax.set_xticks(xpos(fam, values))
    ax.set_xticklabels([fmt_param(fam, v) if np.isclose(TICKED[fam], v).any() else ''
                        for v in values])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel(xlabel)
    end, at = BREAK[fam]
    xb = np.sqrt(at * (values[-2] if np.isinf(end) else values[1]))
    base = ax.get_xaxis_transform()
    ax.plot([xb], [0], transform=base, marker='_', markersize=7, markeredgewidth=4, color=SURFACE,
            clip_on=False, zorder=5)
    for dx in (-2.5, 2.5):
        ax.plot([xb], [0], transform=offset_copy(base, ax.figure, x=dx, units='points'),
                marker=[(-1, -2), (1, 2)], markersize=8, markeredgewidth=0.8, color=AXIS,
                clip_on=False, zorder=6)


def plot_broken(ax, fam, values, y, color, lab):
    '''plot_line over the on-scale values, a dotted link across the axis break, and the off-scale
    end as a lone marker. values must be sorted.'''
    v, y = np.asarray(values, float), np.asarray(y, float)
    x, off = xpos(fam, v), np.isclose(v, BREAK[fam][0])
    plot_line(ax, x[~off], y[~off], color, lab)
    k = np.flatnonzero(off)[0]
    j = k - 1 if k else 1
    ax.plot(x[[j, k]], y[[j, k]], color=color, linestyle=(0, (1, 2.5)), linewidth=1.4,
            dash_capstyle='round')
    ax.plot(x[k:k + 1], y[k:k + 1], color=color, marker='o', markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0, linestyle='none')


def log_y(ax, lo, hi, fmt=lambda v: f"{v:g}"):
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, hi)
