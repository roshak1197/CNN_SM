"""Non-ellipsoidal inserts for the Shepp-Logan phantom (reviewer 2.8 and 2.6), all inside region 2 (water).

Primitives (mm, object coordinates; z along the beam):
  'cil'   finite cylinder, axis along 'y' or 'x': centre, radius r, half-length h
  'caja'  axis-aligned box: centre, half-sizes
  'esf'   sphere: centre, radius
Groups:
  LP-x  vertical fibres (axis y) -> line pairs seen by d_x      (upper right)
  LP-y  horizontal fibres (axis x) -> line pairs seen by d_y    (upper left)
  BOX   box with sharp corners and straight edges               (lower left)
  CD    contrast-detail spheres, 3 sizes x 2 contrasts          (lower right)
"""
import numpy as np
import sl_phantom as S

DEL = {'Air': 3.44e-10, 'PMMA': 3.42e-7, 'Water': 2.94e-7, 'Graphite': 5.847e-7, 'Cellulose': 3.989e-7,
       'Hydroxyapatite': 8.41e-7, 'Alumina': 1.03e-6}
MAT_SL = ['PMMA', 'Water', 'Air', 'Air', 'Graphite', 'Graphite', 'Cellulose', 'Hydroxyapatite', 'Alumina', 'Alumina']


def fibras(eje, c0, d, n, h, mat, grupo):
    """n fibres of diameter d (mm), period 2d, centred at c0 = (x, y); axis 'y' (spread in x) or 'x' (spread in y)."""
    out = []
    for k in range(n):
        off = (k - (n - 1) / 2) * 2 * d
        c = (c0[0] + off, c0[1], 0.0) if eje == 'y' else (c0[0], c0[1] + off, 0.0)
        out.append(dict(tipo='cil', eje=eje, c=np.array(c), r=d / 2, h=h, mat=mat, grupo=grupo, d_um=d * 1e3))
    return out


INSERTOS = (
    fibras('y', (1.05, 1.95), 0.120, 3, 0.30, 'Alumina', 'LP-x') +
    fibras('y', (1.62, 1.95), 0.060, 3, 0.30, 'Alumina', 'LP-x') +
    fibras('x', (-1.25, 2.45), 0.060, 3, 0.30, 'Alumina', 'LP-y') +
    fibras('x', (-1.25, 1.97), 0.120, 3, 0.30, 'Alumina', 'LP-y') +
    [dict(tipo='caja', c=np.array([-1.25, -2.10, 0.0]), hs=np.array([0.30, 0.25, 0.25]), mat='Graphite', grupo='BOX')] +
    [dict(tipo='esf', c=np.array([x, y, 0.0]), r=dd / 2, mat=m, grupo='CD', d_um=dd * 1e3)
     for y, m in ((-1.90, 'Alumina'), (-2.30, 'Cellulose'))
     for x, dd in ((0.75, 0.200), (1.12, 0.120), (1.42, 0.060))]
)


def _cuerda_ins(e, o, u):
    if e['tipo'] == 'esf':
        return S._cuerda(dict(c=e['c'], ax=np.full(3, e['r']), phi=0), o, u)
    lo = e['c'] - (e['hs'] if e['tipo'] == 'caja' else 0); hi = e['c'] + (e['hs'] if e['tipo'] == 'caja' else 0)
    if e['tipo'] == 'caja':
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (lo - o) / u; t2 = (hi - o) / u
        tmin = np.nanmax(np.minimum(t1, t2), 1); tmax = np.nanmin(np.maximum(t1, t2), 1)
        return np.clip(tmax - tmin, 0, None)
    # finite cylinder: circle in the plane normal to the axis, slab along the axis
    ia = 1 if e['eje'] == 'y' else 0; ib = [0, 2] if e['eje'] == 'y' else [1, 2]
    oo = o[ib] - e['c'][ib]; uu = u[:, ib]
    A = (uu ** 2).sum(1); B = 2 * uu @ oo; C = oo @ oo - e['r'] ** 2
    disc = B ** 2 - 4 * A * C; ok = disc > 0
    t1 = np.full(len(u), np.inf); t2 = np.full(len(u), -np.inf)
    t1[ok] = (-B[ok] - np.sqrt(disc[ok])) / (2 * A[ok]); t2[ok] = (-B[ok] + np.sqrt(disc[ok])) / (2 * A[ok])
    s1 = (e['c'][ia] - e['h'] - o[ia]) / u[:, ia]; s2 = (e['c'][ia] + e['h'] - o[ia]) / u[:, ia]
    a = np.maximum(t1, np.minimum(s1, s2)); b = np.minimum(t2, np.maximum(s1, s2))
    return np.clip(b - a, 0, None)


def proyeccion_total(s=4.0, sub=4, con_insertos=True):
    deltas = [DEL[m] for m in MAT_SL]
    pr = S.proyeccion(s, deltas, sub)
    if not con_insertos: return pr
    src = np.array([0, 0, S.ZS])
    f = (np.arange(S.NPIX * sub) + 0.5) / sub - S.NPIX / 2
    X, Y = np.meshgrid(f * S.P, f * S.P)
    d = np.stack([X.ravel(), Y.ravel(), np.full(X.size, S.ZD - S.ZS)], 1); u = d / np.linalg.norm(d, axis=1)[:, None]
    add = np.zeros(len(u))
    for e in INSERTOS:
        add += (DEL[e['mat']] - DEL['Water']) * _cuerda_ins(e, src, u)
    return pr + add.reshape(S.NPIX, sub, S.NPIX, sub).mean((1, 3))


def puntos(e, n, rng):
    if e['tipo'] == 'esf':
        q = rng.normal(size=(n, 3)); q /= np.linalg.norm(q, axis=1)[:, None]; return e['c'] + q * e['r'] * rng.random(n)[:, None] ** (1 / 3)
    if e['tipo'] == 'caja':
        return e['c'] + (rng.random((n, 3)) * 2 - 1) * e['hs']
    ang = rng.random(n) * 2 * np.pi; rr = e['r'] * np.sqrt(rng.random(n)); L = (rng.random(n) * 2 - 1) * e['h']
    p = np.zeros((n, 3)) + e['c']
    if e['eje'] == 'y': p[:, 0] += rr * np.cos(ang); p[:, 2] += rr * np.sin(ang); p[:, 1] += L
    else: p[:, 1] += rr * np.cos(ang); p[:, 2] += rr * np.sin(ang); p[:, 0] += L
    return p


def dentro_ins(e, p):
    if e['tipo'] == 'esf': return ((p - e['c']) ** 2).sum(1) <= e['r'] ** 2
    if e['tipo'] == 'caja': return (abs(p - e['c']) <= e['hs']).all(1)
    ia = 1 if e['eje'] == 'y' else 0; ib = [0, 2] if e['eje'] == 'y' else [1, 2]
    return (((p[:, ib] - e['c'][ib]) ** 2).sum(1) <= e['r'] ** 2) & (abs(p[:, ia] - e['c'][ia]) <= e['h'])


def comprobar(s=4.0, n=20000, margen=0.03, seed=1):
    """Each insert inside ellipsoid 2 (with margin, mm), outside ellipsoids 3..10 and the other inserts, also in projection."""
    rng = np.random.default_rng(seed); E = S.elipsoides(s); msgs = []
    def din(e, p, m=0.0):
        th = np.deg2rad(e['phi']); R = np.array([[np.cos(th), np.sin(th), 0], [-np.sin(th), np.cos(th), 0], [0, 0, 1]])
        q = ((p - e['c']) @ R.T) / (e['ax'] + m); return (q ** 2).sum(1) <= 1
    for i, e in enumerate(INSERTOS):
        p = puntos(e, n, rng)
        if not din(E[1], p, -margen).all(): msgs.append(f'insert {i} ({e["grupo"]}) leaves water')
        # projection test: drop z, compare with the x-y footprint of ellipsoids 3..10 (any z) and other inserts
        pz = p.copy(); 
        for j in range(2, 10):
            fp = E[j].copy(); fp = dict(c=E[j]['c'] * [1, 1, 0], ax=E[j]['ax'] * [1, 1, 1e6], phi=E[j]['phi'])
            if din(fp, pz * [1, 1, 0], margen).any(): msgs.append(f'insert {i} ({e["grupo"]}) overlaps ellipsoid {j+1} in projection')
        for k, f in enumerate(INSERTOS):
            if k != i and dentro_ins(f, p).any(): msgs.append(f'insert {i} touches insert {k}')
    return msgs or ['inserts OK (inside water, no overlap in 3D or in projection)']
