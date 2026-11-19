"""Error metrics of every method against the analytic ground truth, written as plain-text tables.

For each dataset:
  results/<ds>/metrics.csv   method, region, RMSE and mean end-point error (px and um on the detector), and, over the
                             object region, Pearson correlation and slope (amplitude ratio) vs the GT (both components
                             pooled). Single-pair methods: mean and std over all 20 pairs.
  results/<ds>/diagnostic/metrics.csv   correlation / slope with the GT displacement for the synthetic inputs.
Reference row 'd = 0': error of predicting no displacement at all (= rms of the GT signal).
Usage: python metrics.py
"""
import csv, os
import numpy as np
from common import GEOM, RESULTS, agreement, errors, field_mask, load_gt, load_result, path, regions

OBJ = {'sphere': 'sample', 'shepp_logan': 'phantom'}


def methods(ds):
    """name -> list of (dx, dy) (one entry per pair for single-pair methods)."""
    pairs = GEOM[ds]['pairs']
    m = {'CNN, 1 pair': [load_result('results', ds, 'cnn_CNN', f'pair_{k:02d}') for k in pairs]}
    for nw in (1, 15):
        m[f'UMPA, 1 pair ({2 * nw + 1}x{2 * nw + 1})'] = [load_result('results', ds, 'umpa', 'single',
                                                                     f'pair_{k:02d}_Nw{nw}') for k in pairs]
    for f in sorted(os.listdir(path('results', ds, 'umpa'))):
        if f.endswith('_dx.txt') and not f.startswith('N01'):
            n, nw = int(f[1:3]), int(f.split('_Nw')[1].split('_')[0])
            m[f'UMPA, {n} pairs ({2 * nw + 1}x{2 * nw + 1})'] = [load_result('results', ds, 'umpa', f[:-7])]
    return m


def table(ds):
    g = load_gt(ds); gx, gy = g['dx_px'], g['dy_px']
    M = methods(ds)
    mask = field_mask(ds)
    for v in M.values():
        for dx, dy in v:
            mask &= np.isfinite(dx) & np.isfinite(dy)
    REG = regions(ds, g['label'], mask)
    um = GEOM[ds]['um_per_px']
    rows = []
    for name, maps in list(M.items()) + [('d = 0', [(0 * gx, 0 * gy)])]:
        for r, k in REG.items():
            e = [errors(dx, dy, gx, gy, k) for dx, dy in maps]
            ag = [agreement(dx, dy, gx, gy, k) for dx, dy in maps] if (r == OBJ[ds] and name != 'd = 0') else None
            row = dict(method=name, region=r, n_maps=len(maps), n_px=int(k.sum()))
            for q in ('rmse', 'epe'):
                v = np.array([x[q] for x in e])
                row[f'{q}_px'] = v.mean(); row[f'{q}_px_std'] = v.std()
                row[f'{q}_um'] = v.mean() * um; row[f'{q}_um_std'] = v.std() * um
            for q in ('corr', 'slope'):
                row[q] = np.mean([x[q] for x in ag]) if ag else np.nan
                row[q + '_std'] = np.std([x[q] for x in ag]) if ag else np.nan
            rows.append(row)
    with open(path('results', ds, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r_ in rows:
            w.writerow({k: (f'{v:.5g}' if isinstance(v, float) else v) for k, v in r_.items()})
    print(f'\n== {ds}: RMSE / EPE (um on the detector), corr / slope in the {OBJ[ds]}')
    for r_ in rows:
        extra = f"  corr {r_['corr']:+.2f} slope {r_['slope']:+.2f}" if np.isfinite(r_['corr']) else ''
        print(f"{r_['method']:28s} {r_['region']:14s} {r_['rmse_um']:9.3f} / {r_['epe_um']:9.3f}{extra}")
    return rows


def diagnostic(ds):
    g = load_gt(ds); gx, gy = g['dx_px'], g['dy_px']
    d = os.path.join(RESULTS, ds, 'diagnostic')
    obj = regions(ds, g['label'], field_mask(ds))[OBJ[ds]]
    rows = []
    for meth, pre in (('CNN, 1 pair', 'cnn_CNN'), ('UMPA, 1 pair (11x11)', 'umpa_Nw5')):
        for case in ('warp', 'absorption', 'real'):
            dx, dy = load_result('results', ds, 'diagnostic', f'{pre}_{case}')
            k = obj & np.isfinite(dx) & np.isfinite(dy)
            a = agreement(dx, dy, gx, gy, k)
            rows.append(dict(method=meth, input=case, corr=a['corr'], slope=a['slope'],
                             rms_output_um=float(np.sqrt(np.mean(np.r_[dx[k], dy[k]] ** 2))) * GEOM[ds]['um_per_px']))
    with open(os.path.join(d, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r_ in rows:
            w.writerow({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r_.items()})
    print(f'-- {ds} diagnostic (corr / slope with the GT displacement, object region)')
    for r_ in rows:
        print(f"   {r_['method']:22s} {r_['input']:11s} corr {r_['corr']:+.2f} slope {r_['slope']:+.3f}")
    return rows


if __name__ == '__main__':
    for ds in GEOM:
        table(ds)
        diagnostic(ds)
