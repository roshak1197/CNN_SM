"""Analytic ground truth of the Geant4 Shepp-Logan phantom (+ inserts) on the detector grid, written to
data/shepp_logan/ground_truth/*.txt.

Geometry (same as the Geant4 application in geant4/shepp_logan): point source at z = -850 mm, phantom centred at z = 0,
detector at z = +650 mm (M = 1.765, z2 = 650 mm), 256 x 256 pixels of 55 um, 28 keV. The phantom (sl_phantom.py,
sl_insertos.py) is a modified 3D Shepp-Logan of 10 nested ellipsoids, scaled to 4 mm, plus 19 inserts in the water
region (alumina fibres of 120 and 60 um, a graphite box, alumina and cellulose spheres of 200/120/60 um).

int delta dl and int mu dl are computed by exact ray-ellipsoid / ray-cylinder / ray-box intersection along the
cone-beam rays, averaged over SUB x SUB rays per pixel. Displacement d = -(z2/P) d(int delta dl)/dx_obj (px),
x_obj = x_det / M; phase = -(2 pi / lambda) int delta dl.
delta: values of the Geant4 data/*_delta.txt tables at 28 keV (sl_insertos.DEL). mu: total attenuation (xraylib
CS_Total_CP x density); the values are fixed below so that xraylib is not needed (set USE_XRAYLIB = True to recompute).

Files: dx_px, dy_px (physical sign), proj_delta_mm, mu_line, phase_rad, label (0 background, 1..10 ellipsoid regions,
innermost wins, 11 + k = insert k of sl_insertos.INSERTOS).
"""
import os
import numpy as np
import sl_phantom as S
import sl_insertos as I
from common import DATA, GEOM

SCALE, SUB = 4.0, 4
E_KEV = GEOM['shepp_logan']['E_keV']
MU_MM = {'Air': 4.759188289395796e-05, 'PMMA': 0.03919887795214909, 'Water': 0.041752482457798494,  # 1/mm, 28 keV
         'Graphite': 0.06066329936956246, 'Cellulose': 0.050263118279669464,
         'Hydroxyapatite': 0.8018961817946227, 'Alumina': 0.36249967187399573}
USE_XRAYLIB = False


def mu_xraylib():
    import xraylib as xr
    dens = {'PMMA': 1.19, 'Water': 1.0, 'Air': 0.001205, 'Graphite': 2.21, 'Cellulose': 1.42, 'Hydroxyapatite': 3.16,
            'Alumina': 3.961}
    form = {'PMMA': 'C5H8O2', 'Water': 'H2O', 'Graphite': 'C', 'Cellulose': 'C6H10O5',
            'Hydroxyapatite': 'Ca10P6O26H2', 'Alumina': 'Al2O3'}
    return {m: (xr.CS_Total_CP('Air, Dry (near sea level)', E_KEV) if m == 'Air' else xr.CS_Total_CP(form[m], E_KEV))
            * dens[m] / 10 for m in dens}


def rays(sub):
    k = (np.arange(S.NPIX * sub) + 0.5) / sub - 0.5
    X = (k - S.NPIX / 2) * S.P                      # column -> x
    Y = (S.NPIX / 2 - k) * S.P                      # row -> y (row 0 at +y)
    XX, YY = np.meshgrid(X, Y)
    d = np.stack([XX.ravel(), YY.ravel(), np.full(XX.size, S.ZD - S.ZS)], 1)
    return np.array([0, 0, S.ZS]), d / np.linalg.norm(d, axis=1)[:, None]


def line_integral(value, sub=SUB):
    src, u = rays(sub)
    E = S.elipsoides(SCALE)
    tot = np.zeros(len(u))
    for i, e in enumerate(E):
        base = 0.0 if S.PADRE[i] is None else value[I.MAT_SL[S.PADRE[i]]]
        tot += (value[I.MAT_SL[i]] - base) * S._cuerda(e, src, u)
    for e in I.INSERTOS:
        tot += (value[e['mat']] - value['Water']) * I._cuerda_ins(e, src, u)
    return tot.reshape(S.NPIX, sub, S.NPIX, sub).mean((1, 3))


def labels():
    src, u = rays(1)
    lab = np.zeros(len(u), int)
    for i, e in enumerate(S.elipsoides(SCALE)):
        lab[S._cuerda(e, src, u) > 0] = i + 1
    for k, e in enumerate(I.INSERTOS):
        lab[I._cuerda_ins(e, src, u) > 0] = 11 + k
    return lab.reshape(S.NPIX, S.NPIX)


def main():
    mu_tab = mu_xraylib() if USE_XRAYLIB else MU_MM
    proj = line_integral(I.DEL)
    g_row, g_col = np.gradient(proj, S.P / S.M)
    lam = 12.398419843 / E_KEV * 1e-7
    out = dict(dx_px=-g_col * S.Z2 / S.P, dy_px=-g_row * S.Z2 / S.P, proj_delta_mm=proj,
               mu_line=line_integral(mu_tab), phase_rad=-2 * np.pi / lam * proj)
    d = os.path.join(DATA, 'shepp_logan', 'ground_truth')
    for n, a in out.items():
        np.savetxt(os.path.join(d, n + '.txt'), a, fmt='%.9e')
    np.savetxt(os.path.join(d, 'label.txt'), labels(), fmt='%d')
    print(f"max |d| {np.hypot(out['dx_px'], out['dy_px']).max():.4f} px; min T {np.exp(-out['mu_line']).min():.3f}")
    print('\n'.join(S.comprobar_anidamiento(SCALE)), '|', '\n'.join(I.comprobar(SCALE)))


if __name__ == '__main__':
    main()
