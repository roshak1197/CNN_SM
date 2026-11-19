"""Analytic ground truth of the Geant4 Nylon sphere, written to data/sphere/ground_truth/*.txt.

Geometry, determined from the images themselves (chord fit of -ln(sum S / sum R): projected radius 72.80 px, centre
(128, 128)) and consistent with the simulation set-up: sphere radius R = 2 mm (4 mm diameter), source at z = -750 mm,
sphere centre at z = 0, detector at z = +750 mm (M = 2, z2 = 750 mm), 256 x 256 pixels of 55 um, 28 keV.
Material: Geant4 'Nylon' (C6H11NO, 1.4 g/cm3); delta(28 keV) = 3.279e-7 from the Geant4 table data/Nylon_delta.txt
(the value Geant4 uses for refraction), mu = 0.0437 /mm.
Displacement d = -(z2/P) d(int delta dl)/dx_obj (px, physical sign), SUB x SUB rays per pixel.
"""
import os
import numpy as np
from common import DATA

P, NPIX, SUB = 0.055, 256, 4
ZS, ZD = -750.0, 750.0
R_MM, DELTA, MU = 2.0, 3.27893992691118e-07, 0.0437
M, Z2, E_KEV = (ZD - ZS) / (0 - ZS), ZD, 28.0


def chord(sub):
    k = (np.arange(NPIX * sub) + 0.5) / sub - 0.5
    XX, YY = np.meshgrid((k - NPIX / 2) * P, (NPIX / 2 - k) * P)
    u = np.stack([XX, YY, np.full_like(XX, ZD - ZS)], -1)
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    o = np.array([0, 0, ZS])
    b = u @ o; c = o @ o - R_MM ** 2
    disc = b ** 2 - c
    L = np.where(disc > 0, 2 * np.sqrt(np.clip(disc, 0, None)), 0.0)
    return L.reshape(NPIX, sub, NPIX, sub).mean((1, 3))


def main():
    L = chord(SUB)
    proj = DELTA * L
    g_row, g_col = np.gradient(proj, P / M)
    lam = 12.398419843 / E_KEV * 1e-7
    out = dict(dx_px=-g_col * Z2 / P, dy_px=-g_row * Z2 / P, proj_delta_mm=proj, mu_line=MU * L,
               phase_rad=-2 * np.pi / lam * proj)
    d = os.path.join(DATA, 'sphere', 'ground_truth')
    for n, a in out.items():
        np.savetxt(os.path.join(d, n + '.txt'), a, fmt='%.9e')
    np.savetxt(os.path.join(d, 'label.txt'), (chord(1) > 0).astype(int), fmt='%d')
    print(f"M {M:.3f}, z2 {Z2} mm; max |d| {np.hypot(out['dx_px'], out['dy_px']).max():.4f} px")


if __name__ == '__main__':
    main()
