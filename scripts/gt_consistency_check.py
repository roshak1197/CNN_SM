"""Consistency check of the analytic ground truth against the Geant4 refraction.

The 20 sphere references are warped by the exact ground-truth displacement (no intensity change) and UMPA (20 pairs,
3 x 3 window, same settings as run_umpa.py) is applied to them. If the GT matches what Geant4 simulates, UMPA must
recover the same amplitude ratio (slope vs GT) on the warped references as on the real Geant4 sample images; a lower
slope than 1 is then a property of the estimator (sub-pixel bias, window), not of the GT. The test is repeated with the
displacement x 10 to show that the deficit shrinks for larger shifts.
Writes results/sphere/gt_consistency.txt.  Usage: python3 gt_consistency_check.py  (needs UMPA)
"""
import numpy as np
from scipy.ndimage import map_coordinates
from common import agreement, field_mask, load_gt, load_result, load_stack, path, regions
from run_umpa import CONFIG, umpa

ds = 'sphere'
g = load_gt(ds); gx, gy = g['dx_px'], g['dy_px']
R, _ = load_stack(ds)
yy, xx = np.mgrid[0:256, 0:256].astype(float)
obj = regions(ds, g['label'], field_mask(ds))['sample']
lines = []
real = load_result('results', ds, 'umpa', 'N20_Nw1')
k = obj & np.isfinite(real[0])
lines.append(f"real Geant4 samples, UMPA 20 pairs 3x3: slope {agreement(*real, gx, gy, k)['slope']:.3f}")
for f in (1, 10):
    W = np.array([map_coordinates(r, [yy - f * gy, xx - f * gx], order=3, mode='reflect') for r in R])
    dx, dy = umpa(W, R, 1, CONFIG[ds]['df'])
    k = obj & np.isfinite(dx)
    a = agreement(dx, dy, f * gx, f * gy, k)
    lines.append(f"references warped by {f:>2d} x GT, UMPA 20 pairs 3x3: slope {a['slope']:.3f}, corr {a['corr']:.3f}")
open(path('results', ds, 'gt_consistency.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
