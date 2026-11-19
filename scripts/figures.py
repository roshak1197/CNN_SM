"""Figures of the supplementary material, from the .txt data and results (run metrics.py first).

Displacements in micrometres on the detector (d_px x 55 um), phases in radians. Each map figure is written with a
diverging (RdBu_r) and a gray colour map: figures/<name>_{rdbu,gray}.{png,pdf}.
  fig_data_overview       reference, sample, transmission, GT phase, GT d_x, GT d_y for both phantoms
  fig_<ds>_comparison     GT | UMPA 1 pair 3x3 | UMPA 1 pair 31x31 | CNN 1 pair | UMPA 20 pairs 3x3, + RMSE/EPE bars
  fig_sphere_vs_pairs     CNN 1 pair vs UMPA with 1, 5, 10, 15, 20 pairs (3x3): maps and RMSE/EPE bars
  fig_diagnostic          CNN and UMPA on pure speckle displacement / absorption only / real sample
  fig_spectrum            50 kVp W spectrum (SpekPy table of data/spectrum)
Usage: python figures.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from common import FIGURES, GEOM, field_mask, load_gt, load_pair, load_result, path, regions

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8})
CMAPS = {'rdbu': 'RdBu_r', 'gray': 'gray'}
OBJ = {'sphere': 'sample', 'shepp_logan': 'phantom'}
COL = {'UMPA, 1 pair (3x3)': '#2a78d6', 'UMPA, 1 pair (31x31)': '#8bb8ec', 'CNN, 1 pair': '#eb6834',
       'UMPA, 20 pairs (3x3)': '#1baf7a'}
SHORT = {'sphere': 'Sphere', 'shepp_logan': 'Shepp–Logan'}
os.makedirs(FIGURES, exist_ok=True)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIGURES, f'{name}.{ext}'), dpi=250, bbox_inches='tight')
    plt.close(fig)


def read_metrics(ds):
    with open(path('results', ds, 'metrics.csv')) as f:
        return {(r['method'], r['region']): r for r in csv.DictReader(f)}


def method_maps(ds, first=True):
    k = GEOM[ds]['pairs'][0]
    return {'UMPA, 1 pair (3x3)': load_result('results', ds, 'umpa', 'single', f'pair_{k:02d}_Nw1'),
            'UMPA, 1 pair (31x31)': load_result('results', ds, 'umpa', 'single', f'pair_{k:02d}_Nw15'),
            'CNN, 1 pair': load_result('results', ds, 'cnn_CNN', f'pair_{k:02d}'),
            'UMPA, 20 pairs (3x3)': load_result('results', ds, 'umpa', 'N20_Nw1')}


def label(n):
    return n.replace('3x3', '3×3').replace('31x31', '31×31')


def bars(ax, met, names, regs, q, ylabel, cap, col=None):
    col = COL if col is None else col
    x = np.arange(len(regs)); w = 0.8 / len(names)
    for i, n in enumerate(names):
        v = np.array([float(met[(n, r)][f'{q}_um']) for r in regs])
        s = np.array([float(met[(n, r)][f'{q}_um_std']) for r in regs])
        xs = x + (i - (len(names) - 1) / 2) * w
        ax.bar(xs, np.minimum(v, cap), w, color=col.get(n, '0.5'), edgecolor='k', lw=0.4, zorder=3)
        for xi, vi, si in zip(xs, v, s):
            if 0 < si and vi < cap:
                ax.errorbar(xi, vi, si, color='k', lw=0.6, capsize=1.5, zorder=4)
            t = f'↑{vi:.0f}' if vi > cap else f'{vi:.2f}'
            ax.text(xi, min(vi + si, cap) + cap * 0.02, t, ha='center', va='bottom', fontsize=5 if len(regs) > 2 else 5.5,
                    rotation=90 if (len(names) > 4 or len(regs) > 2) else 0, bbox=dict(fc='white', ec='none', pad=0.3), zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([r.replace(' ', '\n') for r in regs], fontsize=7); ax.set_ylim(0, cap * 1.3); ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='0.9', lw=0.6, zorder=0)


def maps_block(fig, gs, cols, mask, um, vmax, cmap):
    for j, (n, (dx, dy)) in enumerate(cols):
        for i, (c, t) in enumerate(((dx, '$d_x$'), (dy, '$d_y$'))):
            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(np.where(mask, c * um, np.nan), cmap=cmap, vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
            if i == 0: ax.set_title(label(n).replace(' (', '\n('), fontsize=7)
            if j == 0: ax.set_ylabel(t + ' (µm)')
    return im


def data_overview():
    for sfx, cmap in CMAPS.items():
        fig, axs = plt.subplots(2, 6, figsize=(7.4, 2.6), gridspec_kw=dict(wspace=0.35, hspace=0.12))
        for i, ds in enumerate(GEOM):
            R, S = load_pair(ds, GEOM[ds]['pairs'][0]); g = load_gt(ds); um = GEOM[ds]['um_per_px']
            Rm = np.mean([load_pair(ds, k)[0] for k in GEOM[ds]['pairs'][:5]], 0)
            T = np.where(Rm > 0, np.mean([load_pair(ds, k)[1] for k in GEOM[ds]['pairs'][:5]], 0) / np.maximum(Rm, 1), np.nan)
            v = np.percentile(np.abs(np.r_[g['dx_px'].ravel(), g['dy_px'].ravel()]), 99.5) * um
            panels = [(R, 'reference R', 'gray', None), (S, 'sample S', 'gray', None),
                      (T, 'transmission (5 pairs)', 'gray', (np.nanpercentile(T, 0.5), 1.02)),
                      (g['phase_rad'], 'GT phase (rad)', 'gray', None),
                      (g['dx_px'] * um, 'GT $d_x$ (µm)', cmap, (-v, v)), (g['dy_px'] * um, 'GT $d_y$ (µm)', cmap, (-v, v))]
            for j, (img, t, cm, lim) in enumerate(panels):
                ax = axs[i, j]
                im = ax.imshow(img, cmap=cm, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
                ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
                if i == 0: ax.set_title(t, fontsize=7)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=5)
            axs[i, 0].set_ylabel(SHORT[ds], fontsize=7)
        save(fig, f'fig_data_overview_{sfx}')


def comparison(ds):
    g = load_gt(ds); gx, gy = g['dx_px'], g['dy_px']; um = GEOM[ds]['um_per_px']
    M = method_maps(ds)
    mask = field_mask(ds)
    for dx, dy in M.values():
        mask &= np.isfinite(dx) & np.isfinite(dy)
    met = read_metrics(ds)
    regs = list(regions(ds, g['label'], mask))
    names = list(M)
    v = np.percentile(np.abs(np.r_[gx[mask], gy[mask]]), 99.5) * um
    cap = 4 * float(met[('d = 0', OBJ[ds])]['rmse_um'])
    for sfx, cmap in CMAPS.items():
        fig = plt.figure(figsize=(7.4, 5.3))
        gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.15], hspace=0.25, wspace=0.06)
        im = maps_block(fig, gs, [('Ground truth', (gx, gy))] + list(M.items()), mask, um, v, cmap)
        fig.colorbar(im, ax=fig.axes[:10], fraction=0.02, pad=0.01, label='d (µm on the detector)')
        sub = gs[2, :].subgridspec(1, 2, wspace=0.25)
        bars(fig.add_subplot(sub[0]), met, names, regs, 'rmse', 'RMSE (µm)', cap)
        bars(fig.add_subplot(sub[1]), met, names, regs, 'epe', 'mean EPE (µm)', cap)
        H = [Patch(facecolor=COL[n], edgecolor='k', lw=0.4, label=label(n)) for n in names]
        fig.legend(handles=H, loc='upper center', ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.075 if len(regs) < 3 else 0.03))
        save(fig, f'fig_{ds}_comparison_{sfx}')


def sphere_vs_pairs():
    ds = 'sphere'; g = load_gt(ds); gx, gy = g['dx_px'], g['dy_px']; um = GEOM[ds]['um_per_px']
    names = ['CNN, 1 pair', 'UMPA, 1 pair (3x3)'] + [f'UMPA, {n} pairs (3x3)' for n in (5, 10, 15, 20)]
    greens = ['#c7e9c0', '#a1d99b', '#74c476', '#31a354', '#006d2c']
    col = dict(COL); col.update(dict(zip(names[1:], greens)))
    M = {'CNN, 1 pair': load_result('results', ds, 'cnn_CNN', 'pair_01')}
    for n in (5, 10, 15, 20):
        M[f'UMPA, {n} pairs (3x3)'] = load_result('results', ds, 'umpa', f'N{n:02d}_Nw1')
    mask = field_mask(ds)
    for dx, dy in M.values():
        mask &= np.isfinite(dx) & np.isfinite(dy)
    met = read_metrics(ds); regs = ['sample', 'background']
    v = np.percentile(np.abs(np.r_[gx[mask], gy[mask]]), 99.5) * um
    cap = 4 * float(met[('d = 0', 'sample')]['rmse_um'])
    for sfx, cmap in CMAPS.items():
        fig = plt.figure(figsize=(7.4, 4.9))
        gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.4], hspace=0.25, wspace=0.06)
        im = maps_block(fig, gs, [('Ground truth', (gx, gy))] + list(M.items()), mask, um, v, cmap)
        fig.colorbar(im, ax=fig.axes[:12], fraction=0.015, pad=0.01, label='d (µm on the detector)')
        sub = gs[2, :].subgridspec(1, 2, wspace=0.25)
        bars(fig.add_subplot(sub[0]), met, names, regs, 'rmse', 'RMSE (µm)', cap, col)
        bars(fig.add_subplot(sub[1]), met, names, regs, 'epe', 'mean EPE (µm)', cap, col)
        H = [Patch(facecolor=col[n], edgecolor='k', lw=0.4, label=label(n)) for n in names]
        fig.legend(handles=H, loc='upper center', ncol=6, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.07))
        save(fig, f'fig_sphere_vs_pairs_{sfx}')


def diagnostic():
    cases = [('warp', 'pure speckle\ndisplacement'), ('absorption', 'absorption only\n(no displacement)'),
             ('real', 'real sample')]
    meths = [('CNN', 'cnn_CNN'), ('UMPA 11×11', 'umpa_Nw5')]
    for sfx, cmap in CMAPS.items():
        fig, axs = plt.subplots(2, 7, figsize=(7.6, 2.6), gridspec_kw=dict(wspace=0.05, hspace=0.35))
        for i, ds in enumerate(GEOM):
            g = load_gt(ds); um = GEOM[ds]['um_per_px']; mask = field_mask(ds)
            obj = regions(ds, g['label'], mask)[OBJ[ds]]
            with open(path('results', ds, 'diagnostic', 'metrics.csv')) as f:
                dm = {(r['method'].split(',')[0], r['input']): r for r in csv.DictReader(f)}
            v = np.percentile(np.abs(g['dx_px'][mask]), 99.5) * um
            axs[i, 0].imshow(np.where(mask, g['dx_px'] * um, np.nan), cmap=cmap, vmin=-v, vmax=v)
            axs[i, 0].set_ylabel(SHORT[ds], fontsize=7)
            if i == 0: axs[i, 0].set_title('GT $d_x$', fontsize=7)
            j = 1
            for mn, pre in meths:
                for c, ct in cases:
                    dx, _ = load_result('results', ds, 'diagnostic', f'{pre}_{c}')
                    ax = axs[i, j]
                    im = ax.imshow(np.where(mask, dx * um, np.nan), cmap=cmap, vmin=-v, vmax=v)
                    r = dm[(mn.split()[0], c)]
                    ax.text(0.03, 0.03, f"r = {float(r['corr']):+.2f}", transform=ax.transAxes, fontsize=5.5,
                            color='k', bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))
                    if i == 0: ax.set_title(f'{mn}:\n{ct}', fontsize=6)
                    j += 1
            cb = fig.colorbar(im, ax=axs[i, :], fraction=0.012, pad=0.01); cb.set_label('$d_x$ (µm)', fontsize=6)
            cb.ax.tick_params(labelsize=5)
        for ax in axs.ravel():
            ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
        save(fig, f'fig_diagnostic_{sfx}')


def spectrum():
    k, phi = np.loadtxt(path('data', 'spectrum', 'spectrum_W50kVp_Be25um_spekpy.txt')).T
    emean = (k * phi).sum() / phi.sum()
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.plot(k, phi / 1e7, color='#2a78d6', lw=1.1)
    ax.axvline(emean, color='#eb6834', ls='--', lw=1.0, label=f'mean energy {emean:.1f} keV')
    ax.set_xlabel('photon energy (keV)'); ax.set_ylabel('fluence (10$^7$ ph cm$^{-2}$ keV$^{-1}$ mAs$^{-1}$ at 1 m)')
    ax.set_title('W anode, 50 kVp, 25 µm Be, 12° (SpekPy)'); ax.set_xlim(0, 51)
    ax.legend(fontsize=7, frameon=False); ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'fig_spectrum')


if __name__ == '__main__':
    data_overview()
    for ds in GEOM:
        comparison(ds)
    sphere_vs_pairs()
    diagnostic()
    spectrum()
    print('figures written to', FIGURES)
