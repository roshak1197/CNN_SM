"""Equal-dose (single-pair) comparison with other methods, six datasets (section S8 of the supplementary material).

Columns: reference | UMPA, 1 pair (31 x 31) | StrainNet-h, 1 pair | ESPINNet, 1 pair (replicated x20) | CNN (this work), 1 pair.
Reference: analytic ground truth (sphere, Shepp-Logan, Star) or UMPA with all pairs, 7 x 7 (MIST 15, fly 20, ESPINNet
data 20). On MIST and the fly the all-pair UMPA is noisy at pixel level (split-half correlation ~0) but reliable after
integration, so the phase metrics are the meaningful ones there.
Metrics (mean over every single pair): displacement correlation and slope (amplitude ratio) with the reference, one
component (d_x; d_y for the Star), inside the object (sphere: r < 85 px; Shepp-Logan: phantom) or the whole interior
(border B excluded, |d_ref| < 10 px); integrated phase (Frankot-Chellappa of d P / z2, x 2 pi P / (M lambda)):
similarity (correlation) and scale (regression slope) against the integrated reference. No output is rescaled. The sign
convention of each method is fixed once per dataset: by the displacement correlation with the GT (simulations, Star) or
by the integrated-phase similarity with the reference (real data, whose displacement reference is noisy).
Writes results/comparison/table.csv, table.md and figures/fig_comparison_<n>_<ds>_{gray,rdbu}.{png,pdf}.
Usage: python comparison.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import FIGURES, GEOM, RESULTS, THIRD, frankot_chellappa, load_gt, path, star_gt

R_ = lambda *p: os.path.join(RESULTS, *p)


def rd(prefix):
    f = prefix + '_dx.txt'
    if not os.path.exists(f):
        f = prefix + '_dx.txt.gz'
    ext = f[len(prefix) + 3:]
    return np.array([np.loadtxt(prefix + '_dx' + ext), np.loadtxt(prefix + '_dy' + ext)])


def method_prefix(ds, m, k):
    if ds in GEOM and m == 'cnn':
        return R_(ds, 'cnn_CNN', f'pair_{k:02d}')
    if ds in GEOM and m == 'umpa31':
        return R_(ds, 'umpa', 'single', f'pair_{k:02d}_Nw15')
    return R_('comparison', ds, m, f'pair_{k:02d}')


ORDER = ['umpa31', 'strainnet', 'espinnet1', 'cnn']
NAMES = {'umpa31': 'UMPA, 1 pair (31×31)', 'strainnet': 'StrainNet, 1 pair', 'espinnet1': 'ESPINNet, 1 pair*',
         'cnn': 'CNN (this work), 1 pair', 'espinnetN': 'ESPINNet, all pairs'}
SHORT = {'umpa31': 'UMPA (31×31)', 'strainnet': 'StrainNet', 'espinnet1': 'ESPINNet*', 'cnn': 'CNN (this work)'}
DATASETS = [('sphere', '1'), ('shepp_logan', '2'), ('mist', '3'), ('fly', '4'), ('star', '5'), ('espinnet', '6')]


def config(ds):
    if ds in GEOM:
        g = GEOM[ds]; gt = load_gt(ds)
        return dict(title=g['name'], pairs=g['pairs'], ref=np.array([gt['dx_px'], gt['dy_px']]), reflab='Analytic GT',
                    B=16, P=g['pixel_mm'], z=g['z2_mm'], E=g['E_keV'], M=g['M'], comp=0, gt=True, label=gt['label'])
    t = THIRD[ds]
    if ds == 'star':
        return dict(title=t['name'], pairs=t['pairs'], ref=np.array(star_gt()), reflab='Analytic GT', B=t['border'],
                    P=None, comp=1, gt=True)
    return dict(title=t['name'], pairs=t['pairs'], ref=rd(R_('comparison', ds, 'reference', 'umpa_all')),
                reflab=t['reference'].replace('7x7', '7×7'), B=t['border'], P=t['pixel_mm'], z=t['z2_mm'],
                E=t['E_keV'], M=t['M'], comp=0, gt=False)


def phase(d, cfg):
    c = (slice(cfg['B'], -cfg['B']),) * 2
    d = [np.clip(np.nan_to_num(a[c]), -10, 10) for a in d]
    lam = 12.398419843e-7 / cfg['E']                                   # mm
    return frankot_chellappa(d[0] * cfg['P'] / cfg['z'], d[1] * cfg['P'] / cfg['z']) * (cfg['P'] / cfg['M']) * 2 * np.pi / lam


sim = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]
scale = lambda a, b: ((a - a.mean()) * (b - b.mean())).sum() / ((b - b.mean()) ** 2).sum()


def analyse(ds):
    cfg = config(ds); ref = np.nan_to_num(cfg['ref']); H, W = ref.shape[1:]
    c = (slice(cfg['B'], -cfg['B']),) * 2
    mk = np.zeros((H, W), bool); mk[c] = True; mk &= np.all(np.abs(ref) < 10, 0)
    if ds == 'sphere':
        mk &= np.hypot(*np.mgrid[-128:128, -128:128]) < 85
    if ds == 'shepp_logan':
        mk &= cfg['label'] > 0
    phref = phase(ref, cfg) if cfg['P'] else None
    out = {'ref': (ref, phref)}; rows = []
    h1 = R_('comparison', ds, 'reference', 'umpa_half1')
    if os.path.exists(h1 + '_dx.txt.gz'):        # reliability of the all-pair UMPA reference: its two halves
        a_, b_ = np.nan_to_num(rd(h1)), np.nan_to_num(rd(R_('comparison', ds, 'reference', 'umpa_half2')))
        i = cfg['comp']; k_ = mk & (np.abs(a_[i]) < 10) & (np.abs(b_[i]) < 10)
        rows.append(dict(dataset=ds, method='reference split-half (UMPA, two halves of the pairs)', n_pairs=2,
                         sign_x=1, sign_y=1, corr=sim(a_[i][k_], b_[i][k_]), corr_std=0.0,
                         slope=np.nan, slope_std=np.nan, phase_similarity=sim(phase(a_, cfg), phase(b_, cfg)),
                         phase_similarity_std=0.0, phase_scale=np.nan))
    methods = ORDER + (['espinnetN'] if os.path.isdir(R_('comparison', ds, 'espinnetN')) else [])
    for m in methods:
        dd = [np.nan_to_num(rd(R_('comparison', ds, 'espinnetN', 'all_pairs')))] if m == 'espinnetN' else \
             [np.nan_to_num(rd(method_prefix(ds, m, k))) for k in cfg['pairs']]
        if cfg['gt']:
            sg = [np.sign(np.mean([sim(d[i][mk], ref[i][mk]) for d in dd])) if ref[i][mk].std() > 0 else 1.0 for i in (0, 1)]
        else:
            sg = max(((a, b) for a in (1, -1) for b in (1, -1)),
                     key=lambda q: np.mean([sim(phase([q[0] * d[0], q[1] * d[1]], cfg), phref) for d in dd[:4]]))
        dd = [np.array([sg[0] * d[0], sg[1] * d[1]]) for d in dd]; i = cfg['comp']
        ok = lambda d: mk & (np.abs(d[i]) < 10)
        C_ = [sim(d[i][ok(d)], ref[i][ok(d)]) for d in dd]
        A_ = [(d[i][ok(d)] * ref[i][ok(d)]).sum() / (ref[i][ok(d)] ** 2).sum() for d in dd]
        if phref is not None:
            PH = [phase(d, cfg) for d in dd]; S_ = [sim(p, phref) for p in PH]; K_ = [scale(p, phref) for p in PH]
        else:
            PH = [None]; S_ = K_ = [np.nan]
        out[m] = (dd[0], PH[0], np.mean(C_), np.mean(A_), np.mean(S_), np.mean(K_))
        rows.append(dict(dataset=ds, method=NAMES[m], n_pairs=len(dd), sign_x=int(sg[0]), sign_y=int(sg[1]),
                         corr=np.mean(C_), corr_std=np.std(C_), slope=np.mean(A_), slope_std=np.std(A_),
                         phase_similarity=np.mean(S_), phase_similarity_std=np.std(S_), phase_scale=np.mean(K_)))
    return cfg, out, rows


def figure(ds, n, cfg, out):
    c = (slice(cfg['B'], -cfg['B']),) * 2; i = cfg['comp']
    um = cfg['P'] * 1e3 if cfg['P'] else 1.0; unit = 'µm' if cfg['P'] else 'px'
    kinds = ['displacement'] + (['phase'] if cfg['P'] else [])
    panels = [(cfg['reflab'], out['ref'])] + [(SHORT[m], out[m]) for m in ORDER]
    for cmap, tag in (('gray', 'gray'), ('RdBu_r', 'rdbu')):
        fig, ax = plt.subplots(len(kinds), 5, figsize=(16, 3.6 * len(kinds) + (1.2 if ds == 'star' else 0)), squeeze=False)
        for r, kind in enumerate(kinds):
            for j, (nm, o) in enumerate(panels):
                a = ax[r, j]
                im = um * o[0][i][c] if kind == 'displacement' else o[1] - o[1].mean()
                v = np.nanpercentile(np.abs(im), 99.5) or 1e-9
                h = a.imshow(im, cmap=cmap, vmin=-v, vmax=v, aspect='auto' if ds == 'star' else 'equal')
                a.set_xticks([]); a.set_yticks([])
                t = nm if r == 0 else ''
                if j > 0:
                    t += ('\n' if t else '') + (f'corr {o[2]:.2f}, slope {o[3]:.2f}' if kind == 'displacement'
                                                else f'sim. {o[4]:.2f}, scale {o[5]:.2f}')
                a.set_title(t, fontsize=12)
                cb = fig.colorbar(h, ax=a, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=9)
                lab = f'$d_{"xy"[i]}$ [{unit}]' if kind == 'displacement' else 'phase [rad]'
                cb.set_label(lab, fontsize=10)
                if j == 0:
                    a.set_ylabel(lab, fontsize=12)
        plt.tight_layout()
        for e in ('png', 'pdf'):
            plt.savefig(os.path.join(FIGURES, f'fig_comparison_{n}_{ds}_{tag}.{e}'), dpi=150)
        plt.close()


def main():
    rows = []
    for ds, n in DATASETS:
        cfg, out, r = analyse(ds); rows += r; figure(ds, n, cfg, out); print(ds, 'done', flush=True)
    os.makedirs(R_('comparison'), exist_ok=True)
    with open(R_('comparison', 'table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r.items()})
    L = ['| Dataset | Method | displacement corr | displacement slope | phase similarity | phase scale |',
         '|---|---|---|---|---|---|']
    for r in rows:
        if 'split-half' in r['method']:
            L.append(f"| {r['dataset']} | {r['method']} | {r['corr']:.2f} | – | {r['phase_similarity']:.2f} | – |"); continue
        ph = '–' if np.isnan(r['phase_similarity']) else f"{r['phase_similarity']:.2f} ± {r['phase_similarity_std']:.2f}"
        sc = '–' if np.isnan(r['phase_scale']) else f"{r['phase_scale']:.2f}"
        L.append(f"| {r['dataset']} | {r['method']} | {r['corr']:.2f} ± {r['corr_std']:.2f} | {r['slope']:.2f} | {ph} | {sc} |")
    open(R_('comparison', 'table.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
