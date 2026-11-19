"""Shepp-Logan (Geant4, 28 keV mono, real SiC 125 um): are the fine structures recovered by the CNN real or artefacts?
(Reviewer R2 2.8.) Per-structure analysis of the inserts and of the small ellipsoids, CNN vs UMPA vs GT.

For every structure s (label from gt_sl_geant4_28keV.npz, 11 + k = INSERTOS[k], 6..10 = small ellipsoids):
  W  = structure mask dilated DIL px (window), T = GT (d_x, d_y) inside W.
  Fit method = a * T + plane (offset + x + y slope, per component) by least squares over W  ->  a = recovered fraction
  of the structure's own (non-planar) signal; a = 1 perfect, 0 nothing. The plane absorbs the gradient of the mother
  ellipsoid, so only the structure itself is tested.
  Null distribution: the SAME template T and window shape are placed at many positions inside GT-flat water (region 2,
  away from every boundary and insert) and fitted to the method there. sigma_a = std(a_null) is the noise of a, and
  z = a / sigma_a the detection significance. mean(a_null) ~ 0 and the tail of a_null measure how often the method
  produces a structure of that shape where there is none (false positives).
Hallucination test (texture): std of the high-pass (x - gauss(x, 3 px)) field in GT-flat phantom interior vs background.
If a method creates object-dependent structure that is not in the GT, the flat-interior texture exceeds the background.
Repeatability: two independent pairs (00, 01; different modulator positions and seeds) -> std(CNN_00 - CNN_01)/sqrt 2.

Inputs: results of run_cnn.py / run_umpa.py (physical sign) and data/shepp_logan/ground_truth. Mask: common.field_mask.
Writes results/shepp_logan/inserts/{inserts_metrics.csv, inserts_summary.txt} and
figures/fig_shepp_logan_inserts.{png,pdf} (maps + profiles, um on the detector) and
figures/fig_shepp_logan_amplitude_vs_material.{png,pdf} (recovered fraction vs mu/delta).
Usage: python inserts_analysis.py
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
import sl_insertos as I
from common import FIGURES, GEOM, RESULTS, field_mask, load_gt, load_result

OUT = os.path.join(RESULTS, 'shepp_logan', 'inserts'); os.makedirs(OUT, exist_ok=True)
URAD = GEOM['shepp_logan']['um_per_px']     # name kept from the original script: here um on the detector per px
DIL = 2
rng = np.random.default_rng(0)

gt = load_gt('shepp_logan')
gt['materiales'] = np.array(I.MAT_SL)
gx, gy, lab = gt['dx_px'], gt['dy_px'], gt['label']
R = lambda *p: load_result('results', 'shepp_logan', *p)
MET = {'CNN 1 pair': R('cnn_CNN', 'pair_00'), 'CNN 1 pair (pair 01)': R('cnn_CNN', 'pair_01'),
       'UMPA 1 pair Nw 1': R('umpa', 'N01_Nw1'), 'UMPA 1 pair Nw 3': R('umpa', 'N01_Nw3'),
       'UMPA 20 pairs Nw 1': R('umpa', 'N20_Nw1'), 'UMPA 20 pairs Nw 3': R('umpa', 'N20_Nw3')}
mask = field_mask('shepp_logan')
for dx, dy in MET.values():
    mask &= np.isfinite(dx) & np.isfinite(dy)
MET = {k: (np.nan_to_num(a), np.nan_to_num(b)) for k, (a, b) in MET.items()}

b = np.zeros_like(mask)
b[:-1] |= lab[:-1] != lab[1:]; b[1:] |= lab[:-1] != lab[1:]
b[:, :-1] |= lab[:, :-1] != lab[:, 1:]; b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
FLAT = mask & (lab == 2) & ~binary_dilation(b, iterations=4) & ~binary_dilation(lab >= 11, iterations=6)
FLAT_IN = mask & (lab > 0) & ~binary_dilation(b, iterations=4) & ~binary_dilation(lab >= 11, iterations=4)
BKG = mask & (lab == 0) & ~binary_dilation(b, iterations=4)
YY, XX = np.mgrid[0:256, 0:256].astype(float)

# structures: inserts (grouped where they act as one test object) and small ellipsoids
EST = []
for k, e in enumerate(I.INSERTOS):
    if e['grupo'] in ('LP-x', 'LP-y'): continue          # fibres are tested as 3-fibre line-pair groups below
    nom = {'LP-x': 'fibre (vertical)', 'LP-y': 'fibre (horizontal)', 'BOX': 'box', 'CD': 'sphere'}[e['grupo']]
    EST.append(dict(nombre=f"{e['grupo']} {e['mat']} {nom} " + (f"{e['d_um']:.0f} um" if 'd_um' in e else '0.6 mm'),
                    grupo=e['grupo'], mat=e['mat'], d=e.get('d_um', 600.0), m=lab == 11 + k))
for g, d in (('LP-x', 120), ('LP-x', 60), ('LP-y', 120), ('LP-y', 60)):  # each 3-fibre line-pair group as one object
    ks = [11 + k for k, e in enumerate(I.INSERTOS) if e['grupo'] == g and e['d_um'] == d]
    EST.append(dict(nombre=f'{g} Alumina 3 fibres {d} um', grupo=g + ' group', mat='Alumina', d=float(d), m=np.isin(lab, ks)))
MATS = gt['materiales']
for j in range(6, 11):
    m = lab == j
    if m.sum():
        EST.append(dict(nombre=f'ellipsoid {j} ({MATS[j - 1]})', grupo='ellipsoid', mat=str(MATS[j - 1]),
                        d=2 * np.sqrt(m.sum() / np.pi) * 55 / 1.765, m=m))


def ajuste(W, Tx, Ty, mx, my):
    """method = a T + plane (per component) over the window pixels; returns a."""
    n = W.sum(); x, y = XX[W] - XX[W].mean(), YY[W] - YY[W].mean(); z = np.zeros(n); o = np.ones(n)
    A = np.block([[Tx[:, None], o[:, None], x[:, None], y[:, None], z[:, None], z[:, None], z[:, None]],
                  [Ty[:, None], z[:, None], z[:, None], z[:, None], o[:, None], x[:, None], y[:, None]]])
    return np.linalg.lstsq(A, np.concatenate([mx, my]), rcond=None)[0][0]


def nulos(W, nmax=300):
    """Shifts that move window W entirely inside FLAT."""
    r, cc = np.nonzero(W); out = []
    for _ in range(20000):
        dr, dc = rng.integers(-256, 256, 2)
        rr, c2 = r + dr, cc + dc
        if rr.min() < 0 or c2.min() < 0 or rr.max() > 255 or c2.max() > 255: continue
        if FLAT[rr, c2].all(): out.append((rr, c2))
        if len(out) >= nmax: break
    return out


filas = []
for s in EST:
    W = binary_dilation(s['m'], iterations=DIL) & mask
    if W.sum() < 4: continue
    Tx, Ty = gx[W], gy[W]
    # GT amplitude of the structure's own signal: residual of GT after removing the plane
    sig = np.sqrt(np.mean(np.concatenate([Tx - Tx.mean(), Ty - Ty.mean()]) ** 2))
    pos = nulos(W); r, cc = np.nonzero(W)
    A3 = np.c_[np.ones(W.sum()), XX[W], YY[W]]
    pr = lambda v: v[W] - A3 @ np.linalg.lstsq(A3, v[W], rcond=None)[0]
    dres, mres = pr(gt['proj_delta_mm']), pr(gt['mu_line'])
    s['mu_delta'] = (dres @ mres) / (dres @ dres)     # local d(int mu)/d(int delta) of the structure (1/mm)
    for nm, (mx, my) in MET.items():
        a = ajuste(W, Tx, Ty, mx[W], my[W])
        an = np.array([ajuste(W, Tx, Ty, mx[rr, c2], my[rr, c2]) for rr, c2 in pos]) if pos else np.array([np.nan])
        sa = np.std(an); fp = np.mean(an >= a) if np.isfinite(a) and len(an) > 1 else np.nan
        filas.append(dict(estructura=s['nombre'], grupo=s['grupo'], material=s['mat'], d_um=round(s['d'], 1),
                          n_px=int(W.sum()), gt_rms_px=sig, mu_over_delta_per_mm=s['mu_delta'], metodo=nm, a=a, sigma_a=sa, z=a / sa,
                          a_null_mean=np.mean(an), a_null_p99=np.percentile(an, 99) if len(an) > 1 else np.nan,
                          p_false=fp, max_abs_z_null=np.max(np.abs(an - an.mean())) / sa, n_null=len(pos)))

with open(os.path.join(OUT, 'inserts_metrics.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader()
    for r_ in filas:
        w.writerow({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r_.items()})

# texture / hallucination test and repeatability
hp = lambda v: v - gaussian_filter(v, 3)
tex = {}
for nm, (mx, my) in list(MET.items()) + [('GT', (gx, gy))]:
    t_in = np.sqrt(np.mean(np.concatenate([hp(mx)[FLAT_IN], hp(my)[FLAT_IN]]) ** 2))
    t_bk = np.sqrt(np.mean(np.concatenate([hp(mx)[BKG], hp(my)[BKG]]) ** 2))
    tex[nm] = (t_in, t_bk)
rep, crep = {}, {}
for R, nr in ((FLAT_IN, 'flat interior'), (BKG, 'background')):
    rep[nr] = np.std(np.concatenate([(MET['CNN 1 pair'][i] - MET['CNN 1 pair (pair 01)'][i])[R] for i in (0, 1)])) / np.sqrt(2)
    crep[nr] = np.corrcoef(np.r_[hp(MET['CNN 1 pair'][0])[R], hp(MET['CNN 1 pair'][1])[R]],
                           np.r_[hp(MET['CNN 1 pair (pair 01)'][0])[R], hp(MET['CNN 1 pair (pair 01)'][1])[R]])[0, 1]

# absorption leakage: does the recovered fraction a depend on the structure's mu/delta (material), not on its size?
fil_ok = lambda f: not (f['material'] == 'Cellulose' and f['grupo'] == 'CD') and not (f['grupo'] == 'CD' and f['d_um'] == 60)
fuga = {}
for nm in MET:
    if nm.startswith('UMPA 1 pair'): continue
    F = [f for f in filas if f['metodo'] == nm and fil_ok(f)]
    x = np.array([f['mu_over_delta_per_mm'] for f in F]); y = np.array([f['a'] for f in F])
    fuga[nm] = (np.corrcoef(x, y)[0, 1], *np.polyfit(x, y, 1)[::-1])

# ---------------- text summary ----------------
L = []
FOC = ['CNN 1 pair', 'CNN 1 pair (pair 01)', 'UMPA 1 pair Nw 1', 'UMPA 20 pairs Nw 1', 'UMPA 20 pairs Nw 3']
L.append('Recovered fraction a (sigma_a) [z]   per structure; a = 1 is the GT amplitude, z = a / sigma_a from null placements')
L.append('structure'.ljust(44) + ''.join(m[:22].rjust(24) for m in FOC))
for s in EST:
    rr_ = [f for f in filas if f['estructura'] == s['nombre']]
    if not rr_: continue
    L.append(s['nombre'][:44].ljust(44) + ''.join(
        (lambda f: f"{f['a']:6.2f} ({f['sigma_a']:.2f}) [{f['z']:5.1f}]".rjust(24))(next(f for f in rr_ if f['metodo'] == m)) for m in FOC))
L.append('')
L.append('High-pass texture rms (px): GT-flat phantom interior / background  (object-induced texture if interior >> background)')
for nm, (a1, a2) in tex.items():
    L.append(f'  {nm:26s} {a1:.4f} / {a2:.4f}   ratio {a1 / a2:.2f}')
L.append('')
L.append('CNN repeatability, pairs 00 vs 01 (independent modulator positions and Poisson noise):')
for nr in rep:
    L.append(f'  {nr:14s} noise std(diff)/sqrt2 = {rep[nr]:.4f} px, correlation of the high-pass fields = {crep[nr]:+.2f}')
L.append('')
L.append('Recovered fraction a vs mu/delta of the structure (inserts + ellipsoids 6-10, detected ones: no cellulose / 60 um spheres)')
for nm, (cr, a0, a1) in fuga.items():
    L.append(f'  {nm:26s} corr {cr:+.2f}   a = {a0:.2f} + {a1 * 1e6:.2f}e-6 * mu/delta[1/mm]')
L.append('')
L.append('Null placements (template in GT-flat water): max |z| reached, per method (false-positive level)')
for nm in FOC:
    L.append(f"  {nm:26s} max |z_null| = {max(f['max_abs_z_null'] for f in filas if f['metodo'] == nm):.1f}  "
             f"(n placements per structure ~{int(np.median([f['n_null'] for f in filas if f['metodo'] == nm]))})")
txt = '\n'.join(L); print(txt)
open(os.path.join(OUT, 'inserts_summary.txt'), 'w').write(txt + '\n')

# ---------------- figure ----------------
plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.titlesize': 8})
fig = plt.figure(figsize=(7.4, 9.6))
gs = fig.add_gridspec(4, 4, hspace=0.55, wspace=0.4, height_ratios=[1, 1, 1, 1.1])
COLS = [('GT', (gx, gy)), ('CNN, 1 pair', MET['CNN 1 pair']), ('UMPA, 1 pair', MET['UMPA 1 pair Nw 1']),
        ('UMPA, 20 pairs', MET['UMPA 20 pairs Nw 1'])]
COL = {'GT': 'k', 'CNN, 1 pair': '#eb6834', 'UMPA, 1 pair': '#2a78d6', 'UMPA, 20 pairs': '#1baf7a'}


def caja(ms, pad=4):
    r, cc = np.nonzero(ms); return r.min() - pad, r.max() + pad + 1, cc.min() - pad, cc.max() + pad + 1


ZON = [('LP-x', 0, caja(np.isin(lab, [11 + k for k, e in enumerate(I.INSERTOS) if e['grupo'] == 'LP-x']))),
       ('LP-y', 1, caja(np.isin(lab, [11 + k for k, e in enumerate(I.INSERTOS) if e['grupo'] == 'LP-y']))),
       ('CD', 0, caja(np.isin(lab, [11 + k for k, e in enumerate(I.INSERTOS) if e['grupo'] == 'CD'])))]
V = 0.03
for i, (g, comp, (a0, a1, b0, b1)) in enumerate(ZON):
    for j, (nm, (mx, my)) in enumerate(COLS):
        ax = fig.add_subplot(gs[i, j]); img = (mx, my)[comp][a0:a1, b0:b1]
        sc = 1.0                  # all methods on the same scale (CNN recovers 0.7-1.1 of the insert amplitude)
        im = ax.imshow(img * sc * URAD, cmap='RdBu_r', norm=TwoSlopeNorm(0, -V * URAD, V * URAD),
                       extent=[b0 - 0.5, b1 - 0.5, a1 - 0.5, a0 - 0.5])
        ax.set_title(f'{g}: {nm}' + (' (×3)' if sc != 1 else ''), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(('d$_x$' if comp == 0 else 'd$_y$') + ' (µm)')
    cbx = fig.colorbar(im, ax=fig.axes[-4:], fraction=0.02, pad=0.01); cbx.ax.tick_params(labelsize=6)

# profiles: LP-x (d_x averaged over fibre length), LP-y (d_y averaged over fibre length)
bot = gs[3, :].subgridspec(1, 2, wspace=0.3)
ax1 = fig.add_subplot(bot[0]); ax2 = fig.add_subplot(bot[1])
for ax, g, comp in ((ax1, 'LP-x', 0), (ax2, 'LP-y', 1)):
    a0, a1, b0, b1 = [z for gg, _, z in ZON if gg == g][0]
    lb = np.isin(lab, [11 + k for k, e in enumerate(I.INSERTOS) if e['grupo'] == g])
    if comp == 0:
        rr = np.nonzero(lb.any(1))[0]; sl = (slice(rr.min() + 2, rr.max() - 1), slice(b0, b1)); axis = 0; xs_ = np.arange(b0, b1)
    else:
        cc_ = np.nonzero(lb.any(0))[0]; sl = (slice(a0, a1), slice(cc_.min() + 2, cc_.max() - 1)); axis = 1; xs_ = np.arange(a0, a1)
    for nm, f in COLS:
        if nm == 'UMPA, 1 pair': continue
        # no amplitude scaling: each curve in urad, only a linear trend fitted to that curve is removed
        p = f[comp][sl].mean(axis); p = p - np.polyval(np.polyfit(xs_, p, 1), xs_)
        ax.plot(xs_, p * URAD, color=COL[nm], lw=1.1, label=nm)
    ax.axhline(0, color='0.6', lw=0.5)
    ax.set_xlabel('column (px)' if comp == 0 else 'row (px)'); ax.set_ylabel('d (µm), trend removed', labelpad=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(f'{g}: ' + ('mean d$_x$ along vertical fibres' if comp == 0 else 'mean d$_y$ along horizontal fibres'), fontsize=7)
ax1.legend(fontsize=5.5, frameon=True, framealpha=0.9, loc='upper left')
# panel letters: one per row of maps, one per profile
filas_ax = [fig.axes[4 * i + i] for i in range(3)]          # first map of each row (colorbars interleave)
for ax, t in zip(filas_ax + [ax1, ax2], 'abcde'):
    ax.text(-0.12 if ax in (ax1, ax2) else -0.32, 1.12, f'({t})', transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(FIGURES, f'fig_shepp_logan_inserts.{ext}'), dpi=250, bbox_inches='tight')

# ---------------- separate figure: recovered fraction vs mu/delta (material) ----------------
fig2, ax3 = plt.subplots(figsize=(3.6, 2.9))
for nm, col, mk in (('CNN 1 pair', '#eb6834', 'o'), ('UMPA 20 pairs Nw 1', '#1baf7a', 's')):
    F = [f for f in filas if f['metodo'] == nm and fil_ok(f)]
    x = np.array([f['mu_over_delta_per_mm'] for f in F]) / 1e5
    ax3.errorbar(x, [f['a'] for f in F], yerr=[f['sigma_a'] for f in F], fmt=mk, ms=3.5, color=col, lw=0.8,
                 label=f"{'CNN, 1 pair' if nm.startswith('CNN') else 'UMPA, 20 pairs'} (r = {fuga[nm][0]:+.2f})")
    xx = np.linspace(0, x.max() * 1.05, 10); ax3.plot(xx, fuga[nm][1] + fuga[nm][2] * xx * 1e5, color=col, lw=0.7, ls='--')
    if nm.startswith('CNN'):
        for f_, xv in zip(F, x):
            if f_['material'] in ('Graphite', 'Cellulose', 'Hydroxyapatite'):
                ax3.annotate(f_['material'] + (' box' if f_['grupo'] == 'BOX' else ''), (xv, f_['a']),
                             xytext=(5, -3), textcoords='offset points', fontsize=6, color='0.25')
ax3.annotate('alumina\n(fibres, spheres,\nellipsoids)', (4.3, 1.45), fontsize=6, color='0.25', ha='center')
ax3.axhline(1, color='k', lw=0.6, ls=':'); ax3.set_xlim(0, None)
ax3.set_xlabel('μ/δ of the structure (10$^5$ mm$^{-1}$)'); ax3.set_ylabel('recovered fraction a (1 = GT)')
ax3.legend(fontsize=6, frameon=False, loc='lower right'); ax3.spines[['top', 'right']].set_visible(False)
for ext in ('png', 'pdf'):
    fig2.savefig(os.path.join(FIGURES, f'fig_shepp_logan_amplitude_vs_material.{ext}'), dpi=250, bbox_inches='tight')
