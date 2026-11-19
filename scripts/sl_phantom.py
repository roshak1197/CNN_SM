"""Shepp-Logan 3D phantom (modified, Yu-Ye-Wang / phantom3d) for the CNN validation, and its analytic ground truth.

Geometry of ~/CNN_Sphere_Sample/PEPI2: point-like source at z = -85 cm, object centred at z = 0, detector at z = +65 cm
(M = 150/85, z2 = 0.65 m), 256 x 256 pixels of 55 um. Ellipsoids are nested: 1 contains 2, 2 contains 3..10, which are
disjoint (checked by `comprobar_anidamiento`). Each region has one material; the ground truth is the projected
refractive index decrement, int delta dl, by exact ray-ellipsoid intersection along the cone-beam rays.

Units: mm. Output displacements in detector pixels: d = alpha * z2 / P, alpha = d(int delta dl)/dx_object.
"""
import numpy as np

ZS, ZD = -850.0, 650.0          # source and detector z (mm); object at z = 0
P, NPIX = 0.055, 256            # detector pixel (mm), pixels
M = (ZD - ZS) / (0 - ZS)        # 1.7647
Z2 = ZD                          # object-detector distance (mm)

# modified Shepp-Logan 3D: (value, a, b, c, x0, y0, z0, phi_deg)  [phi = rotation about z]
SL = [(1.0, .6900, .920, .810, 0, 0, 0, 0),
      (-.8, .6624, .874, .780, 0, -.0184, 0, 0),
      (-.2, .1100, .310, .220, .22, 0, 0, -18),
      (-.2, .1600, .410, .280, -.22, 0, 0, 18),
      (.1, .1600, .250, .410, 0, .35, -.15, 0),   # a .21 -> .16: disjoint from 4 in 3D (Geant4 needs it)
      (.1, .0460, .046, .050, 0, .1, .25, 0),
      (.1, .0460, .046, .050, 0, -.1, .25, 0),
      (.1, .0460, .023, .050, -.08, -.605, 0, 0),
      (.1, .0230, .023, .020, 0, -.606, 0, 0),
      (.1, .0230, .046, .020, .06, -.605, 0, 0)]
PADRE = [None, 0, 1, 1, 1, 1, 1, 1, 1, 1]   # mother region of each ellipsoid


def elipsoides(s):
    """Ellipsoids scaled by s (mm): list of dicts with centre, semi-axes, rotation (deg about z)."""
    return [dict(c=np.array([x * s, y * s, z * s]), ax=np.array([a * s, b * s, cc * s]), phi=ph)
            for (_, a, b, cc, x, y, z, ph) in SL]


def _cuerda(e, o, u):
    """Chord length of rays o + t u (o: (3,), u: (n,3) unit) through ellipsoid e."""
    th = np.deg2rad(e['phi']); R = np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
    oo = R @ (o - e['c']) / e['ax']; uu = (u @ R.T) / e['ax']
    A = (uu ** 2).sum(1); B = 2 * uu @ oo; C = oo @ oo - 1
    disc = B ** 2 - 4 * A * C
    L = np.zeros(len(u)); ok = disc > 0
    L[ok] = np.sqrt(disc[ok]) / A[ok]          # t2 - t1, u is unit so this is a length
    return L


def proyeccion(s, deltas, sub=4):
    """int delta dl (mm * delta) on the detector grid, sub x sub rays per pixel. deltas: 10 region deltas (absolute)."""
    E = elipsoides(s); src = np.array([0, 0, ZS])
    f = (np.arange(NPIX * sub) + 0.5) / sub - NPIX / 2
    X, Y = np.meshgrid(f * P, f * P)
    d = np.stack([X.ravel(), Y.ravel(), np.full(X.size, ZD - ZS)], 1); u = d / np.linalg.norm(d, axis=1)[:, None]
    tot = np.zeros(len(u))
    for i, e in enumerate(E):
        base = 0.0 if PADRE[i] is None else deltas[PADRE[i]]
        tot += (deltas[i] - base) * _cuerda(e, src, u)
    return tot.reshape(NPIX, sub, NPIX, sub).mean((1, 3))    # rows = y, cols = x


def desplazamiento(proj):
    """Displacement (px) from the projected delta: alpha = grad(int delta)/dx_obj, dx_obj = P/M."""
    gy, gx = np.gradient(proj, P / M)
    return gx * Z2 / P, gy * Z2 / P


def fase(proj, E_keV=28.0):
    lam = 12.398419843 / E_keV * 1e-7     # mm
    return -2 * np.pi / lam * proj         # rad


def comprobar_anidamiento(s, n=200000, seed=0):
    """Monte Carlo check: every point of ellipsoid i (i >= 1) lies in its mother, and 2..9 are disjoint."""
    E = elipsoides(s); rng = np.random.default_rng(seed)
    def dentro(e, p):
        th = np.deg2rad(e['phi']); R = np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
        q = ((p - e['c']) @ R.T) / e['ax']; return (q ** 2).sum(1) <= 1
    msgs = []
    for i, e in enumerate(E):
        if PADRE[i] is None: continue
        q = rng.normal(size=(n, 3)); q /= np.linalg.norm(q, axis=1)[:, None]; q *= rng.random(n)[:, None] ** (1 / 3)
        th = np.deg2rad(e['phi']); R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        p = (q * e['ax']) @ R.T + e['c']
        if not dentro(E[PADRE[i]], p).all(): msgs.append(f'{i+1} sale de {PADRE[i]+1}')
        for j in range(2, 10):
            if j != i and i >= 2 and dentro(E[j], p).any(): msgs.append(f'{i+1} toca {j+1}')
    return msgs or ['anidamiento OK']
