import glob
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image

from generator import create_new, calibrate_density
from engine import trophic_coherence
from simulation import P as sim

# Toy-sized GIFs of how GPPM structure responds to each generator parameter,
# one parameter swept per GIF with the others held fixed. Meant to show the
# model's general behavior, so density is calibrated for the toy N itself
# rather than taken from a run.

# --- parameters ---
params = {
    'N': 25,
    'B_FRACTION': sim.B_FRACTION,     # held fixed in the T_hat and c sweeps
    'T_HAT': 0.5,                     # held fixed in the b and c sweeps
    'T_HAT_RANGE': sim.T_HAT_RANGE,   # the tested range
    'B_RANGE': np.arange(1, 11),      # basal counts B; swept as b = B/N
    'C_EXTRA': 0.6,                   # added to the calibrated c (see below)
    'C_MAX': 4.0,                     # c sweep runs from tree-only up to this
    'N_C_FRAMES': 10,
    'N_SEEDS': 200,                   # draws per value when picking a frame
    'FRAME_MS': 1000,
    'HOLD_MS': 2500,                  # last frame, so the loop point is visible
}
P = SimpleNamespace(**params)

# One density for every GIF, calibrated at the toy size, then bumped by
# C_EXTRA: the calibrated c = 1.40 leaves only 12 excess links, too few for
# T_hat to act on -- 4 of the 9 T_hat steps redrew the identical graph. At
# +0.6 (50 edges) every step changes it. The b sweep holds this c fixed too:
# recalibrating per b blows up at this N (calibrate_density's B*ln(B)/N
# margin gives c ~ 8.9 at B = 10, i.e. 222 edges on 25 nodes).
c_default = calibrate_density(P.B_FRACTION, P.N) + P.C_EXTRA

# --- style, after the GPPM paper's figure ---
BG = '#2e3440'
INK = '#e5e9f0'
MUTED = '#8f98a8'
EDGE = '#9aa3b5'
cmap = LinearSegmentedColormap.from_list(
    'gppm', ['#d9a8f5', '#f77ff0', '#f7909a', '#f7ab1f', '#c9cc00'])
NODE_SIZE = 260

save_dir = 'figures/toy'


def n_basal(b):
    return max(1, round(b * P.N))


def draw(b, c, T_hat, seed):
    G = create_new(P.N, b, c, T_hat, np.random.default_rng(seed))[0]
    tc = trophic_coherence(G)
    return G, tc['trophic_levels'], tc['trophic_incoherence']


def tree_x(b, seed):
    '''
    Horizontal position of every node, taken from the tree phase alone.

    create_new draws the whole tree before it touches T_hat or c, so a graph
    built from the same seed with c = (N-B)/N (no excess links) is exactly the
    tree that seed grows its excess links on top of. Laid out tidy-tree style
    (nodes nobody attached to get consecutive slots in depth-first order,
    every other node sits at the mean of the nodes that attached to it), then
    ranked by (position, level) so every node gets its own column -- otherwise
    a chain of single attachments stacks in one column and overlaps whenever
    the excess links pull its trophic levels together.
    '''
    B = n_basal(b)
    tree = create_new(P.N, b, (P.N - B) / P.N, P.T_HAT,
                      np.random.default_rng(seed))[0]
    x, slot = {}, 0

    def place(v):
        nonlocal slot
        above = sorted(tree.predecessors(v))
        if not above:
            x[v] = float(slot)
            slot += 1
            return
        for u in above:
            place(u)
        x[v] = np.mean([x[u] for u in above])

    for root in range(B):   # create_new numbers the basal nodes 0..B-1
        place(root)
    order = sorted(x, key=lambda n: (x[n], tree.nodes[n]['level']))
    mid = (len(order) - 1) / 2
    return tree, {n: rank - mid for rank, n in enumerate(order)}


def sweep(values, settings, same_tree):
    '''
    One graph per sweep value; settings(v) -> (b, c, T_hat).

    A single draw at this N is noisy, so every value is drawn under N_SEEDS
    seeds and the frame shows a representative one: the draw whose q is
    closest to the median q at that value. With same_tree, one seed serves
    the whole sweep -- the one whose q stays closest to the median across all
    values -- so the tree stays put and only the excess links change.
    '''
    q = np.array([[draw(*settings(v), seed)[2] for v in values]
                  for seed in range(P.N_SEEDS)])
    q_med = np.median(q, axis=0)
    if same_tree:
        best = int(np.argmin(((q - q_med) ** 2).sum(axis=1)))
        seeds = [best] * len(values)
    else:
        seeds = np.argmin(np.abs(q - q_med), axis=0)

    frames = []
    for v, seed, med in zip(values, seeds, q_med):
        b, c, T_hat = settings(v)
        G, s, q_frame = draw(b, c, T_hat, seed)
        tree, x = tree_x(b, seed)
        assert set(tree.edges) <= set(G.edges), 'tree differs from the frame'
        frames.append(dict(v=v, b=b, c=c, T_hat=T_hat, G=G, x=x, s=s,
                           q=q_frame, q_med=med))
    return frames


def render_frame(fig, frames, k, s_top, x_half, title, held):
    f = frames[k]
    G, s, x = f['G'], f['s'], f['x']
    fig.clf()
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.80])
    ax.set_facecolor(BG)
    ax.axis('off')

    pos = {n: (x[n], s[n]) for n in G}
    nodes = list(range(P.N))
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=EDGE, width=1.0, alpha=0.75, arrows=True,
        arrowstyle='-|>', arrowsize=9, node_size=NODE_SIZE,
        connectionstyle='arc3,rad=0.12')
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=nodes, node_size=NODE_SIZE,
        node_color=cmap(Normalize(1, s_top)(s[nodes])),
        edgecolors='#eceff4', linewidths=0.8)

    # s axis: arrow up the left side, integer ticks
    x_axis = -x_half - 1.0
    ax.annotate('', xy=(x_axis, s_top + 0.55), xytext=(x_axis, 0.55),
                arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.8))
    for lvl in range(1, int(np.floor(s_top)) + 1):
        ax.text(x_axis - 0.35, lvl, str(lvl), color=INK, fontsize=15,
                ha='right', va='center')
    ax.text(x_axis + 0.3, s_top + 0.5, 'trophic level $s$', color=INK,
            fontsize=15, va='center')
    ax.set_xlim(x_axis - 1.2, x_half + 0.8)
    ax.set_ylim(0.4, s_top + 0.8)

    # header: swept value and q on the left, held parameters under it
    fig.text(0.04, 0.925, title(f), color=INK, fontsize=20, va='center')
    fig.text(0.04, 0.855,
             f'q = {f["q"]:.2f}  (median of {P.N_SEEDS} draws: '
             f'{f["q_med"]:.2f})    {G.number_of_edges()} edges    {held}',
             color=MUTED, fontsize=11, va='center')

    # scrubber: where this frame sits in the sweep
    sx = fig.add_axes([0.62, 0.88, 0.34, 0.06])
    sx.set_facecolor(BG)
    sx.axis('off')
    vals = [fr['v'] for fr in frames]
    sx.plot(vals, np.zeros(len(vals)), color=MUTED, lw=1, zorder=1)
    sx.scatter(vals, np.zeros(len(vals)), s=18, color=BG, edgecolors=MUTED,
               zorder=2)
    sx.scatter([f['v']], [0], s=60, color=INK, zorder=3)
    sx.text(vals[0], -0.9, f'{vals[0]:g}', color=MUTED, fontsize=9,
            ha='center', va='top')
    sx.text(vals[-1], -0.9, f'{vals[-1]:g}', color=MUTED, fontsize=9,
            ha='center', va='top')
    sx.set_ylim(-1.8, 1)


def make_gif(param, frames, title, held):
    '''
    Write figures/toy/<param>/toy_<param>.gif plus every frame as a PNG
    named by its position and swept value (e.g. 03_T_hat_0p40.png, same
    'p'-for-'.' labels as simulation.py's result files).
    '''
    out_dir = os.path.join(save_dir, param)
    os.makedirs(out_dir, exist_ok=True)
    # clear frames from a previous run, which may have swept different values
    for old in glob.glob(os.path.join(out_dir, '*.png')):
        os.remove(old)

    s_top = max(f['s'].max() for f in frames)
    x_half = max(max(abs(v) for v in f['x'].values()) for f in frames)

    fig = plt.figure(figsize=(10, 6.5), dpi=100)
    images = []
    for k, f in enumerate(frames):
        render_frame(fig, frames, k, s_top, x_half, title, held)
        fig.canvas.draw()
        images.append(Image.fromarray(
            np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        stem = f'{k:02d}_{param}_{f["v"]:.2f}'.replace('.', 'p')
        images[-1].save(os.path.join(out_dir, f'{stem}.png'))
    plt.close(fig)

    path = os.path.join(out_dir, f'toy_{param}.gif')
    durations = [P.FRAME_MS] * (len(images) - 1) + [P.HOLD_MS]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=durations, loop=0)
    print(f'saved {path} + {len(images)} frames')
    print('  frame q: ', ' '.join(f'{f["q"]:.2f}' for f in frames))
    print('  median q:', ' '.join(f'{f["q_med"]:.2f}' for f in frames))


def main():
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
    print(f'N={P.N}, held c = {c_default:.3f} = calibrated '
          f'{c_default - P.C_EXTRA:.3f} + {P.C_EXTRA} '
          f'(L = {round(c_default * P.N)}) at b={P.B_FRACTION}')

    # 1) T_hat: same tree every frame, only the excess links are redrawn
    frames = sweep(P.T_HAT_RANGE,
                   lambda T_hat: (P.B_FRACTION, c_default, T_hat),
                   same_tree=True)
    make_gif('T_hat', frames,
             title=lambda f: fr'$\hat{{T}}$ = {f["T_hat"]:.2f}',
             held=f'N = {P.N},  b = {P.B_FRACTION:g},  c = {c_default:.2f}')

    # 2) b: a new tree per frame (B changes), density held at c_default
    frames = sweep(P.B_RANGE / P.N,
                   lambda b: (b, c_default, P.T_HAT),
                   same_tree=False)
    make_gif('b', frames,
             title=lambda f: f'b = {f["b"]:.2f}   (B = {n_basal(f["b"])})',
             held=fr'N = {P.N},  $\hat{{T}}$ = {P.T_HAT:g},  c = {c_default:.2f}')

    # 3) c: same tree every frame, from no excess links up to C_MAX
    c_tree = (P.N - n_basal(P.B_FRACTION)) / P.N
    frames = sweep(np.linspace(c_tree, P.C_MAX, P.N_C_FRAMES),
                   lambda c: (P.B_FRACTION, c, P.T_HAT),
                   same_tree=True)
    make_gif('c', frames,
             title=lambda f: f'c = {f["c"]:.2f}',
             held=fr'N = {P.N},  b = {P.B_FRACTION:g},  $\hat{{T}}$ = {P.T_HAT:g}')


if __name__ == '__main__':
    main()
