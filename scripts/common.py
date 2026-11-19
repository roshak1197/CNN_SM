"""Shared definitions for every script of this repository: paths, geometry, I/O of the .txt data, masks, regions,
metrics and units. Every script reads the plain-text files of data/ and writes plain-text results to results/.

Conventions
-----------
* Images are 256 x 256 arrays in the orientation of the Geant4 detector: pixel (row i, col j) has its centre at
  x = (j - 128) P, y = (128 - i) P (P = 55 um). Rows grow with -y.
* d_x = displacement of the speckle along +column, d_y = along +row, in detector pixels, PHYSICAL sign
  (d = -(z2/P) d(int delta dl)/dx_obj: a ray bends towards lower delta). All result files use this sign.
  Raw UMPA output has the opposite sign (checked against the ground truth on both phantoms); run_umpa.py multiplies
  it by UMPA_SIGN = -1 before writing. The CNN output already has the physical sign.
* Figures show displacements in micrometres on the detector (d_px x P) and phases in radians.
  Refraction angle, if needed: alpha = d P / z2.
"""
import os
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(ROOT, 'figures')
CNN_DIR = os.path.join(ROOT, 'cnn')
UMPA_SIGN = -1
FMT = '%.7e'

# Geometry of each Geant4 dataset (point source, monochromatic 28 keV, ideal photon-counting detector 256 x 256, 55 um)
GEOM = {
    'sphere': dict(name='Nylon sphere', pixel_mm=0.055, z1_mm=750.0, z2_mm=750.0, E_keV=28.0,
                   pairs=list(range(1, 21))),          # files R01..R20 / S01..S20
    'shepp_logan': dict(name='Shepp-Logan phantom', pixel_mm=0.055, z1_mm=850.0, z2_mm=650.0, E_keV=28.0,
                        pairs=list(range(0, 20))),     # files R00..R19 / S00..S19 (Geant4 run numbers)
}
for g in GEOM.values():
    g['M'] = (g['z1_mm'] + g['z2_mm']) / g['z1_mm']
    g['um_per_px'] = g['pixel_mm'] * 1e3
    g['urad_per_px'] = g['pixel_mm'] / g['z2_mm'] * 1e6


def path(*p):
    return os.path.join(ROOT, *p)


def load_pair(ds, k):
    """Reference (no sample) and sample image of modulator position k, float64 counts (double-counted, see README)."""
    d = os.path.join(DATA, ds, 'pairs')
    return np.loadtxt(os.path.join(d, f'R{k:02d}.txt')), np.loadtxt(os.path.join(d, f'S{k:02d}.txt'))


def load_stack(ds, pairs=None):
    pairs = GEOM[ds]['pairs'] if pairs is None else pairs
    R, S = zip(*(load_pair(ds, k) for k in pairs))
    return np.array(R), np.array(S)


def load_gt(ds):
    d = os.path.join(DATA, ds, 'ground_truth')
    g = {n: np.loadtxt(os.path.join(d, n + '.txt')) for n in ('dx_px', 'dy_px', 'proj_delta_mm', 'mu_line', 'phase_rad')}
    g['label'] = np.loadtxt(os.path.join(d, 'label.txt')).astype(int)
    return g


def save_map(a, *p):
    f = path(*p)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    np.savetxt(f, a, fmt=FMT)
    return f


def load_result(*p):
    """Displacement pair (d_x, d_y) written by run_cnn.py / run_umpa.py: <prefix>_dx.txt, <prefix>_dy.txt."""
    f = path(*p)
    return np.loadtxt(f + '_dx.txt'), np.loadtxt(f + '_dy.txt')


def pad_to_full(a, n=256):
    """UMPA crops the border (window + search range): pad with NaN back to n x n, centred."""
    p0, p1 = (n - a.shape[0]) // 2, (n - a.shape[1]) // 2
    return np.pad(a, ((p0, n - a.shape[0] - p0), (p1, n - a.shape[1] - p1)), constant_values=np.nan)


def field_mask(ds, border=8, erode=8):
    """Pixels used for every metric: 8 px border removed and the dark corners outside the 0.006 rad beam cone
    (mean reference < 0.8 x median), eroded by 8 px so that the UMPA window never touches them."""
    Rm = np.mean([np.loadtxt(os.path.join(DATA, ds, 'pairs', f'R{k:02d}.txt')) for k in GEOM[ds]['pairs']], 0)
    m = np.zeros(Rm.shape, bool)
    m[border:-border, border:-border] = True
    return m & binary_erosion(Rm > 0.8 * np.median(Rm), iterations=erode)


def boundaries(lab):
    b = np.zeros(lab.shape, bool)
    b[:-1] |= lab[:-1] != lab[1:]; b[1:] |= lab[:-1] != lab[1:]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]; b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b


def regions(ds, lab, mask):
    """Evaluation regions. Sphere: sample (inside the projected sphere) and background (> 3 px outside).
    Shepp-Logan: phantom, edges (2 px around region boundaries, inserts excluded), inserts (dilated 2 px),
    flat interior, background (label 0, away from edges)."""
    if ds == 'sphere':
        return {'sample': mask & (lab > 0), 'background': mask & ~binary_dilation(lab > 0, iterations=3)}
    edge, ins = binary_dilation(boundaries(lab), iterations=2), binary_dilation(lab >= 11, iterations=2)
    return {'phantom': mask & (lab > 0), 'edges': mask & edge & ~ins, 'inserts': mask & ins,
            'flat interior': mask & (lab > 0) & ~edge & ~ins, 'background': mask & (lab == 0) & ~edge}


def errors(dx, dy, gx, gy, k):
    """RMSE per component combined as sqrt((RMSE_x^2 + RMSE_y^2)/2) and mean end-point error, in px, over mask k."""
    ex, ey = dx[k] - gx[k], dy[k] - gy[k]
    return dict(rmse=float(np.sqrt((np.mean(ex ** 2) + np.mean(ey ** 2)) / 2)), epe=float(np.mean(np.hypot(ex, ey))))


def agreement(dx, dy, gx, gy, k):
    """Pearson correlation and least-squares slope (through the origin) of the method vs the GT, both components
    pooled, over mask k. Slope 1 = correct amplitude."""
    m, g = np.r_[dx[k], dy[k]], np.r_[gx[k], gy[k]]
    return dict(corr=float(np.corrcoef(m, g)[0, 1]), slope=float((m @ g) / (g @ g)))
