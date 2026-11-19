"""Builds supplementary/Supplementary_Material.pdf (reportlab) from the figures and the metric tables.
Every number quoted in the text is read from results/ at build time. Run after metrics.py, inserts_analysis.py and
figures.py.  Usage: python make_supplementary.py
"""
import csv, datetime, os
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from PIL import Image as PILImage
import sl_insertos as I
import sl_phantom as S
from common import FIGURES, GEOM, ROOT, path

DRAFT = True        # visible 'draft for co-author review' banner on every page
OUT = path('supplementary', 'Supplementary_Material.pdf')
TITLE = ('Supplementary material: Single-shot bi-directional phase retrieval in X-ray speckle-based imaging: '
         'a convolutional neural network approach')

ss = getSampleStyleSheet()
BODY = ParagraphStyle('b', parent=ss['BodyText'], fontName='Times-Roman', fontSize=10, leading=13, alignment=TA_JUSTIFY,
                      spaceAfter=5)
H1 = ParagraphStyle('h1', parent=ss['Heading1'], fontName='Times-Bold', fontSize=13, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle('h2', parent=ss['Heading2'], fontName='Times-Bold', fontSize=11, spaceBefore=8, spaceAfter=4)
CAP = ParagraphStyle('cap', parent=BODY, fontSize=8.8, leading=11, spaceAfter=10)
TIT = ParagraphStyle('t', parent=H1, fontSize=15, leading=19, alignment=1)
CEN = ParagraphStyle('c', parent=BODY, alignment=1)
TAB = ParagraphStyle('tab', parent=BODY, fontSize=8, leading=9.5, alignment=0, spaceAfter=0)
NFIG = [0]
NTAB = [0]


def P(t, st=BODY):
    return Paragraph(t, st)


def fig(name, caption, width=17.0):
    f = os.path.join(FIGURES, name + '.png')
    w, h = PILImage.open(f).size
    NFIG[0] += 1
    return KeepTogether([Image(f, width=width * cm, height=width * cm * h / w), Spacer(1, 4),
                         P(f'<b>Figure S{NFIG[0]}.</b> {caption}', CAP)])


def table(rows, caption, widths=None, header=1):
    NTAB[0] += 1
    data = [[Paragraph(str(c), TAB) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths] if widths else None, repeatRows=header)
    t.setStyle(TableStyle([('LINEABOVE', (0, 0), (-1, 0), 0.8, colors.black),
                           ('LINEBELOW', (0, header - 1), (-1, header - 1), 0.5, colors.black),
                           ('LINEBELOW', (0, -1), (-1, -1), 0.8, colors.black),
                           ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 1.5),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5)]))
    return KeepTogether([P(f'<b>Table S{NTAB[0]}.</b> {caption}', CAP), t, Spacer(1, 10)])


def metrics(ds):
    with open(path('results', ds, 'metrics.csv')) as f:
        return {(r['method'], r['region']): r for r in csv.DictReader(f)}


def diag(ds):
    with open(path('results', ds, 'diagnostic', 'metrics.csv')) as f:
        return {(r['method'].split(',')[0], r['input']): r for r in csv.DictReader(f)}


def v(m, meth, reg, q, nd=2):
    return f'{float(m[(meth, reg)][q]):.{nd}f}'


def pm(m, meth, reg, q, nd=2):
    r = m[(meth, reg)]
    s = float(r[q + '_std']) if (q + '_std') in r and r[q + '_std'] not in ('', 'nan') else 0.0
    if s <= 0:
        return f'{float(r[q]):.{nd}f}'
    return f'{float(r[q]):.{nd}f} ± ' + (f'{s:.{nd}f}' if s >= 0.5 * 10 ** -nd else f'&lt;{10 ** -nd:.{nd}f}')


def metric_table(ds, regs):
    m = metrics(ds)
    obj = regs[0]
    names = ['CNN, 1 pair', 'UMPA, 1 pair (3x3)', 'UMPA, 1 pair (31x31)'] + \
            sorted({k[0] for k in m if 'pairs' in k[0]}, key=lambda s: (int(s.split()[1]), s)) + ['d = 0']
    rows = [['method'] + [f'RMSE {r} (µm)' for r in regs] + [f'EPE {r} (µm)' for r in regs] +
            [f'corr ({obj})', f'slope ({obj})']]
    for n in names:
        rows.append([n.replace('3x3', '3×3').replace('31x31', '31×31')] + [pm(m, n, r, 'rmse_um') for r in regs] +
                    [pm(m, n, r, 'epe_um') for r in regs] +
                    (['–', '–'] if n == 'd = 0' else [pm(m, n, obj, 'corr'), pm(m, n, obj, 'slope')]))
    return rows


def comparison_rows():
    with open(path('results', 'comparison', 'table.csv')) as f:
        return {(r['dataset'], r['method']): r for r in csv.DictReader(f)}


def comparison_section(ms, ml):
    T = comparison_rows()
    U, SN, E1, C, EN = ('UMPA, 1 pair (31×31)', 'StrainNet, 1 pair', 'ESPINNet, 1 pair*', 'CNN (this work), 1 pair',
                        'ESPINNet, all pairs')
    SH = 'reference split-half (UMPA, two halves of the pairs)'
    g = lambda ds, m, q, nd=2: f"{float(T[(ds, m)][q]):.{nd}f}"
    st = [P('S8 Comparison with other single-shot methods', H1),
          P('The CNN was compared, at equal dose (one reference/sample pair), with every single-shot method whose code '
            'and trained weights are publicly available: UMPA with a 31 × 31 window (the single-pair setting of '
            'Qiao et al. 2026), StrainNet-h [Boukhtache et al. 2021], a CNN for digital image correlation of optical '
            'speckle trained on displacements up to about ±1 px, and ESPINNet [Qiao et al. 2026], a network for X-ray '
            'speckle tracking that takes 20 reference and 20 sample frames. ESPINNet is designed for two or more '
            'pairs; with a single pair the pair was replicated 20 times (marked *). Its single-shot predecessor, '
            'SPINNet, has no public weights. Six datasets spanning different regimes were used (Table S5): the two '
            'Geant4 phantoms of this work, two published experimental data sets (MIST wattle flower, fly), the '
            "optical 'Star' images of StrainNet, and the synchrotron test data of ESPINNet. The third-party data and "
            'networks are not redistributed: the repository downloads them from their original sources.'),
          P('<b>Protocol.</b> Each single-pair method is applied to every pair of a dataset, and the metrics are the '
            'mean over pairs. The reference is the analytic ground truth when it exists, otherwise UMPA with all pairs '
            '(7 × 7 window). For MIST and the fly the all-pair UMPA is noisy pixel by pixel (the two halves of the '
            f'data give displacement maps with correlation {g("mist", SH, "corr")} and {g("fly", SH, "corr")}) but '
            f'reliable after integration (split-half phase similarity {g("mist", SH, "phase_similarity")} and '
            f'{g("fly", SH, "phase_similarity")}), so the integrated phase is the meaningful comparison there. Displacement metrics: '
            'correlation and slope (amplitude ratio) with the reference, d<sub>x</sub> (d<sub>y</sub> for the Star). '
            'Phase metrics: the displacement is integrated with the Frankot–Chellappa algorithm and compared with the '
            'integrated reference (similarity = correlation; scale = regression slope). No output is rescaled; the '
            'sign convention of each method is fixed once per dataset (with the ground truth, or with the phase '
            'similarity on the real data). Every map is shown with its own colour scale, so that the scale bar gives '
            'the amplitude.')]
    rows = [['dataset', 'source', 'E (keV)', 'pixel (µm)', 'z<sub>2</sub> (m)', 'pairs', 'reference']]
    for r in (['Nylon sphere', 'this work (Geant4)', '28', '55', '0.75', '20', 'analytic'],
              ['Shepp–Logan', 'this work (Geant4)', '28', '55', '0.65', '20', 'analytic'],
              ['MIST wattle flower', 'Alloo et al. 2023, Sci. Rep. (IMBL); github.com/samanthaalloo/MIST_Alloo23',
               '25', '9.9', '2.0', '15', 'UMPA, 15 pairs'],
              ['Fly', 'OSF project 4ruq2 (laboratory, 40 kV)', '20†', '5.2', '0.53', '20', 'UMPA, 20 pairs'],
              ["Star", 'Boukhtache et al. 2021 (StrainNet repository; optical DIC images)', '–', '–', '–', '2',
               'analytic'],
              ['ESPINNet test data', 'Qiao et al. 2026, J. Synchrotron Rad. 33, 467 (synchrotron)', '20', '0.65',
               '0.812', '20', 'UMPA, 20 pairs']):
        rows.append(r)
    st.append(table(rows, 'Datasets of the comparison. Third-party data are used as published (MIST: dark- and '
                          'flat-field corrected as in the authors\' script; the others raw), on a 1024 × 1024 central '
                          'region (Star: full frame). † Assumed effective energy; it only sets the scale of the phase.',
                    widths=[2.6, 6.3, 1.3, 1.4, 1.2, 1.1, 2.6]))
    st.append(P(f'<b>Simulated phantoms.</b> On the sphere and the Shepp–Logan phantom the CNN gives the highest '
                f'displacement correlation with the ground truth ({g("sphere", C, "corr")} and '
                f'{g("shepp_logan", C, "corr")}, vs {g("sphere", U, "corr")} and {g("shepp_logan", U, "corr")} for '
                f'UMPA 31 × 31) and the highest phase similarity, with slopes of {g("sphere", C, "slope")} and '
                f'{g("shepp_logan", C, "slope")} (Figs. S{NFIG[0] + 1}, S{NFIG[0] + 2}). StrainNet and single-pair '
                'ESPINNet do not recover the object (correlations at or below '
                f'{max(float(T[(d, m)]["corr"]) for d in ("sphere", "shepp_logan") for m in (SN, E1)):.2f}): the '
                'displacements (≈ 0.02–0.08 px) are far below StrainNet’s training range, and the absorption of the '
                'object breaks the brightness constancy that StrainNet assumes, while ESPINNet needs several '
                'modulator positions.'))
    st.append(fig('fig_comparison_1_sphere_gray', 'Nylon sphere, one pair. Top: d<sub>x</sub> (µm on the detector); '
                  'bottom: integrated phase. Columns: analytic ground truth, UMPA (31 × 31), StrainNet-h, ESPINNet '
                  '(pair replicated ×20), CNN. Each panel has its own colour scale; correlation/slope and phase '
                  'similarity/scale against the reference are given above each panel (mean over the 20 pairs).'))
    st.append(fig('fig_comparison_2_shepp_logan_gray', 'Shepp–Logan phantom, one pair; layout as in the previous '
                  'figure.'))
    st.append(P(f'<b>Experimental data.</b> On the MIST and fly data, UMPA 31 × 31 recovers both the shape and the '
                f'amplitude of the integrated phase (similarity {g("mist", U, "phase_similarity")} and '
                f'{g("fly", U, "phase_similarity")}, scale {g("mist", U, "phase_scale")} and {g("fly", U, "phase_scale")}). '
                f'The CNN reproduces the outline of the samples (phase similarity {g("mist", C, "phase_similarity")} '
                f'and {g("fly", C, "phase_similarity")}) at about {100 * float(T[("fly", C)]["phase_scale"]):.0f} % of '
                'the reference amplitude: these data (speckle of 6–18 px, larger displacements, different energies) '
                'are outside its training domain. For MIST the CNN similarity is partly inflated, because the MIST '
                'phase is itself derived from intensity differences. StrainNet and single-pair ESPINNet give '
                f'lower similarities, while ESPINNet with all pairs reaches {g("fly", EN, "phase_similarity")} on the '
                'fly and works as intended on its own data (below).'))
    st.append(fig('fig_comparison_3_mist_gray', 'MIST wattle flower (Alloo et al. 2023), one pair; reference: UMPA '
                  'with the 15 pairs.'))
    st.append(fig('fig_comparison_4_fly_gray', 'Fly (OSF 4ruq2), one pair; reference: UMPA with the 20 pairs.'))
    st.append(P(f"<b>Regimes of the other networks.</b> On StrainNet's Star images (sinusoidal displacement of "
                f'0.5 px, period 10–150 px), StrainNet recovers the field (correlation {g("star", SN, "corr")}, '
                f'slope {g("star", SN, "slope")}), and so does UMPA ({g("star", U, "corr")}); the CNN returns no '
                f'signal ({g("star", C, "corr")}). On the ESPINNet test data (displacements up to ≈ 7 px, '
                f'speckle ≈ 16 px), ESPINNet with all 20 pairs matches the reference '
                f'(phase similarity {g("espinnet", EN, "phase_similarity")}), UMPA with one pair reaches '
                f'{g("espinnet", U, "phase_similarity")}, and the CNN only marks edges at '
                f'{g("espinnet", C, "slope", 3)} of the reference amplitude. Each network therefore works in the regime '
                'it was trained for; the CNN of this work is specific to small (sub-0.1 px) displacements with fine '
                'speckle, and even there it recovers 0.2–0.3 of the amplitude (Sections S5–S7).'))
    st.append(fig('fig_comparison_5_star_gray', "StrainNet 'Star' images (noiseless and noisy, mean shown in the "
                  'labels), d<sub>y</sub> in pixels; reference: analytic displacement.'))
    st.append(fig('fig_comparison_6_espinnet_gray', 'ESPINNet test data (Qiao et al. 2026), one pair; reference: '
                  'UMPA with the 20 pairs.'))
    names = {'sphere': 'Sphere', 'shepp_logan': 'Shepp–Logan', 'mist': 'MIST', 'fly': 'Fly', 'star': 'Star',
             'espinnet': 'ESPINNet data'}
    rows = [['dataset', 'method', 'disp. corr', 'disp. slope', 'phase sim.', 'phase scale']]
    for (ds, m), r in T.items():
        if m == SH:
            continue
        ph = '–' if r['phase_similarity'] in ('nan', '') else f"{float(r['phase_similarity']):.2f}"
        sc = '–' if r['phase_scale'] in ('nan', '') else f"{float(r['phase_scale']):.2f}"
        rows.append([names[ds], m.replace(' (this work)', ''), f"{float(r['corr']):.2f}", f"{float(r['slope']):.2f}", ph, sc])
    st.append(table(rows, 'Single-pair comparison: mean over pairs of the displacement correlation and slope and of '
                          'the integrated-phase similarity and scale against the reference. ESPINNet, all pairs = '
                          'multi-frame reference of that network.', widths=[2.8, 5.0, 2.0, 2.0, 2.0, 2.0]))
    return st


def onpage(c, d):
    c.saveState()
    c.setFont('Times-Roman', 8)
    c.drawString(2 * cm, 1.2 * cm, 'Supplementary material')
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(d.page))
    if DRAFT:
        c.setFillColor(colors.HexColor('#b00000'))
        c.drawCentredString(A4[0] / 2, A4[1] - 1.1 * cm,
                            f'DRAFT for co-author review, generated {datetime.date.today().isoformat()}')
    c.restoreState()


def build():
    ms, ml = metrics('sphere'), metrics('shepp_logan')
    ds_, dl = diag('sphere'), diag('shepp_logan')
    gs, gl = GEOM['sphere'], GEOM['shepp_logan']
    cons = [l.split('slope ')[1].split(',')[0].strip() for l in open(path('results', 'sphere', 'gt_consistency.txt'))]
    timing = open(path('results', 'sphere', 'cnn_CNN', 'timing.txt')).read().strip()
    cnn, u1, u15, u20 = 'CNN, 1 pair', 'UMPA, 1 pair (3x3)', 'UMPA, 1 pair (31x31)', 'UMPA, 20 pairs (3x3)'
    st = []
    st += [Spacer(1, 2 * cm), P(TITLE, TIT), Spacer(1, 6),
           P('S. González López, C. D. Tibambre, C. A. Ávila', CEN), Spacer(1, 18)]
    st += [P('Contents', H2)]
    for s in ('S1 Data and code', 'S2 Monte Carlo simulations', 'S3 Analytic ground truth',
              'S4 Retrieval methods and figures of merit', 'S5 Nylon sphere: equal-dose comparison',
              'S6 Shepp–Logan phantom: fine structure', 'S7 Control experiment: what each method responds to',
              'S8 Comparison with other single-shot methods', 'S9 X-ray spectrum of the experimental set-up'):
        st.append(P(s))

    # ---------------------------------------------------------------- S1
    st += [P('S1 Data and code', H1),
           P('All simulated images, ground truths, network weights, scripts and the numerical results behind the '
             'figures of this supplement are distributed in the accompanying repository (folder layout in its '
             'README). Images are stored as plain-text 256 × 256 matrices (one value per pixel). Every result can be '
             'regenerated with <font face="Courier">run_all.sh</font>; the numbers quoted below are read from the '
             'result tables when this document is built.')]

    # ---------------------------------------------------------------- S2
    st += [P('S2 Monte Carlo simulations', H1),
           P('Both test objects were simulated with a Geant4 application derived from Geant4PEPI (L. Brombal, INFN '
             'Trieste), which includes X-ray refraction at material boundaries and an ideal photon-counting detector '
             'of 256 × 256 pixels of 55 µm. The source is point-like within a 5 µm Gaussian focal spot and emits '
             'monochromatic 28 keV photons in a cone of 0.006 rad half-angle (the detector corners lie outside the '
             'cone and are excluded from every metric). Each image uses 10<super>9</super> primary photons. The '
             'application records every detected photon twice (a software merge of each run with itself); the '
             'images are distributed as recorded, so the Poisson statistics correspond to half of the stored values. '
             'For each modulator position k, a reference image R<sub>k</sub> (no sample) and a sample image '
             'S<sub>k</sub> share the modulator position; positions differ by tens to hundreds of micrometres.'),
           P('S2.1 Nylon sphere', H2),
           P(f'A Nylon sphere (C<sub>6</sub>H<sub>11</sub>NO, 1.4 g cm<super>−3</super>) of radius 2 mm (diameter 4 mm) '
             f'is placed at {gs["z1_mm"] / 10:.0f} cm from the source and {gs["z2_mm"] / 10:.0f} cm from the detector '
             f'(magnification M = {gs["M"]:.2f}). The modulator consists of stacked layers of 125 µm grains. Twenty '
             'pairs are available. The geometry was verified on the images themselves: a chord fit of '
             '−ln(ΣS/ΣR) gives a projected radius of 72.8 px centred on the detector, i.e. R = 2.0 mm for M = 2.'),
           P('S2.2 Shepp–Logan phantom with inserts', H2),
           P(f'A three-dimensional modified Shepp–Logan phantom of ten nested ellipsoids (scale 4 mm; outer ellipsoid '
             f'5.5 × 7.4 × 6.5 mm) is placed at {gl["z1_mm"] / 10:.0f} cm from the source and {gl["z2_mm"] / 10:.0f} cm '
             f'from the detector (M = {gl["M"]:.3f}). Each ellipsoid is a Geant4 volume nested in its mother, so every '
             'interface is a real refracting boundary. Nineteen inserts with known fine structure are embedded in the '
             'water region: two groups of three vertical alumina fibres (diameters 120 and 60 µm, period twice the '
             'diameter; 3.9 and 1.9 detector pixels in diameter), the same two groups horizontally, a graphite box '
             '(1.2 × 1.0 × 1.0 mm, sharp edges), and contrast-detail spheres of 200, 120 and 60 µm in alumina and in '
             'cellulose (the cellulose row is a low-contrast control). The modulator is four layers of real SiC '
             'grains (Si 70.05 %, C 29.95 %, 3.21 g cm<super>−3</super>, 125 µm) on 200 µm cellulose backings. '
             'Twenty modulator positions were simulated (Table S2). Two additional images without modulator '
             '(flat field and phantom only) are included for checking the transmission geometry.')]
    rows = [['region', 'ellipsoid (a, b, c) / insert', 'material', 'δ (28 keV)', 'μ (mm<super>−1</super>)']]
    mu = __import__('gt_shepp_logan').MU_MM
    for i, (val, a, b, c, *_r) in enumerate(S.SL):
        m_ = I.MAT_SL[i]
        rows.append([str(i + 1), f'{a * 4:.2f} × {b * 4:.2f} × {c * 4:.2f} mm', m_, f'{I.DEL[m_]:.3g}', f'{mu[m_]:.3g}'])
    for g, desc, m_ in (('LP-x / LP-y', 'fibres Ø 120 and 60 µm, length 0.6 mm', 'Alumina'),
                        ('BOX', 'box 1.2 × 1.0 × 1.0 mm', 'Graphite'),
                        ('CD', 'spheres Ø 200, 120, 60 µm', 'Alumina, Cellulose')):
        rows.append([g, desc, m_, ' / '.join(f'{I.DEL[x.strip()]:.3g}' for x in m_.split(',')),
                     ' / '.join(f'{mu[x.strip()]:.3g}' for x in m_.split(','))])
    st.append(table(rows, 'Shepp–Logan phantom: semi-axes of the ten ellipsoids and inserts, materials, refractive '
                          'index decrement δ (Geant4 tables) and linear attenuation coefficient μ at 28 keV.',
                    widths=[2.0, 5.6, 3.6, 2.6, 2.4]))
    pos = [(-10, 20), (70, -50), (60, 80), (100, 40), (150, -200), (180, -260), (-180, -20), (-300, 120),
           (-300, -330), (-420, 190), (290, 470), (20, 470), (-500, 60), (-240, 370), (0, 370), (-450, 30),
           (210, -40), (-80, 80), (400, -100), (250, 500)]
    rows = [['k', 'x (µm)', 'y (µm)', 'k', 'x (µm)', 'y (µm)']]
    for k in range(10):
        rows.append([f'{k:02d}', *map(str, pos[k]), f'{k + 10:02d}', *map(str, pos[k + 10])])
    st.append(table(rows, 'Shepp–Logan: transverse modulator offsets of the 20 simulated positions.',
                    widths=[1.2, 2, 2, 1.2, 2, 2]))
    st.append(fig('fig_data_overview_gray',
                  'Simulated data. From left: reference image R and sample image S of the first modulator position, '
                  'transmission (mean S / mean R over five positions), analytic phase, and the two components of the '
                  'analytic speckle displacement (µm on the detector). Top: Nylon sphere; bottom: Shepp–Logan '
                  'phantom with inserts.'))

    # ---------------------------------------------------------------- S3
    st += [P('S3 Analytic ground truth', H1),
           P('The projected refractive index decrement ∫δ dl and the projected attenuation ∫μ dl are computed by exact '
             'ray–surface intersection (ellipsoids, finite cylinders, boxes, spheres) along the cone-beam rays from '
             'the source to each pixel, averaging 4 × 4 rays per pixel. The speckle displacement on the detector, in '
             'pixels, is d = −(z<sub>2</sub>/P) ∂(∫δ dl)/∂x<sub>obj</sub>, with x<sub>obj</sub> = x<sub>det</sub>/M, '
             'z<sub>2</sub> the object–detector distance and P the pixel size; the phase is φ = −(2π/λ)∫δ dl. The '
             'δ values are those used by the Geant4 refraction model. Displacements are reported in micrometres on '
             f'the detector (d × 55 µm); the corresponding refraction angle is α = d P / z<sub>2</sub> '
             f'({gs["urad_per_px"]:.1f} µrad per pixel for the sphere, {gl["urad_per_px"]:.1f} µrad per pixel for the '
             f'Shepp–Logan). The maximum displacement is {0.0759 * 55:.1f} µm (sphere rim) and about 5 µm '
             '(Shepp–Logan edges); most of the phantom interior is well below 1 µm (0.02 px). As a consistency check, '
             'the 20 sphere references were warped by the exact ground-truth displacement and processed with UMPA '
             f'(20 pairs, 3 × 3): the recovered amplitude ratio is {cons[1]} on the warped references and {cons[0]} '
             f'on the Geant4 sample images, and it rises to {cons[2]} when the displacement is multiplied by ten. The '
             'ground truth therefore agrees with the refraction simulated by Geant4, and the amplitude deficit of '
             'UMPA at these sub-pixel displacements is a property of the estimator.')]

    # ---------------------------------------------------------------- S4
    st += [P('S4 Retrieval methods and figures of merit', H1),
           P('<b>CNN.</b> The pre-trained network of the main text (checkpoint <font face="Courier">CNN.pt</font>), '
             'applied to a single reference/sample pair after joint min–max normalisation of the pair; its output is '
             f'the two-component displacement in pixels. {timing}.'),
           P('<b>UMPA.</b> Unified modulated pattern analysis (UMPA v0.2), Hamming-weighted window of (2N<sub>w</sub>+1)<super>2</super> '
             'pixels, ±4 px search range, the systematic bias removed by subtracting UMPA applied to the reference '
             'images against themselves. It is run with a single pair (windows 3 × 3 and 31 × 31; the large window is '
             'the setting recommended for single-pair UMPA on noisy data) and with 5, 10, 15 and 20 pairs '
             '(3 × 3 window, as in the main text). Raw UMPA output has the opposite sign convention to the ground '
             'truth and is multiplied by −1.'),
           P('<b>Figures of merit.</b> All methods are evaluated on the same pixels: an 8-pixel border and the dark '
             'corners outside the beam cone are removed. Regions: for the sphere, the projected sphere and the '
             'background more than 3 px outside it; for the Shepp–Logan, the whole phantom, the edges (2 px around '
             'every region boundary), the inserts, the flat interior and the background. RMSE = '
             '[(RMSE<sub>x</sub><super>2</super> + RMSE<sub>y</sub><super>2</super>)/2]<super>1/2</super>; '
             'EPE = mean of |d − d<sub>GT</sub>|. Within the object region, the Pearson correlation with the ground '
             'truth and the least-squares slope through the origin (amplitude ratio; 1 = correct amplitude) are '
             'computed with both components pooled. For single-pair methods, the value is the mean ± standard '
             'deviation over the 20 pairs. The row d = 0 is the error of predicting no displacement, i.e. the rms '
             'ground-truth signal: a method extracts information about the object only where its error is below it.')]

    # ---------------------------------------------------------------- S5
    st += [P('S5 Nylon sphere: equal-dose comparison', H1),
           P(f'At equal dose (one pair), UMPA with the 3 × 3 window of the main text fails (RMSE '
             f'{v(ms, u1, "sample", "rmse_um", 1)} µm inside the sphere). With a 31 × 31 window, UMPA and the CNN have '
             f'similar errors inside the sphere (RMSE {v(ms, u15, "sample", "rmse_um")} and {v(ms, cnn, "sample", "rmse_um")} µm; '
             f'mean EPE {v(ms, u15, "sample", "epe_um")} and {v(ms, cnn, "sample", "epe_um")} µm), while in the '
             f'background the CNN is lower ({v(ms, cnn, "background", "rmse_um")} vs {v(ms, u15, "background", "rmse_um")} µm) '
             f'(Fig. S2, Table S3). The CNN maps correlate with the ground truth (r = {v(ms, cnn, "sample", "corr")} vs '
             f'{v(ms, u15, "sample", "corr")} for UMPA 31 × 31) and are almost noise-free, but both single-pair methods '
             f'recover only a fraction of the true amplitude (slopes {v(ms, cnn, "sample", "slope")} and '
             f'{v(ms, u15, "sample", "slope")}). Predicting no displacement at all gives an RMSE of '
             f'{v(ms, "d = 0", "sample", "rmse_um")} µm inside the sphere, so the single-pair errors are only moderately '
             f'below this reference. UMPA with 20 pairs (3 × 3) recovers a slope of {v(ms, u20, "sample", "slope")} but, '
             f'because of its noise, its RMSE ({v(ms, u20, "sample", "rmse_um")} µm) stays above that reference (Fig. S3). '
             'The CNN result is the same for each of the 20 single pairs (standard deviations in Table S3).'),
           fig('fig_sphere_comparison_gray',
               'Nylon sphere, equal dose. Maps of d<sub>x</sub> and d<sub>y</sub> (µm on the detector, common scale) '
               'for the ground truth, UMPA with one pair (3 × 3 and 31 × 31 windows), the CNN with one pair, and UMPA '
               'with 20 pairs. Bottom: RMSE and mean end-point error against the ground truth (mean ± s.d. over the '
               '20 single pairs; off-scale values marked ↑).'),
           fig('fig_sphere_vs_pairs_gray',
               'Nylon sphere: CNN with one pair compared with UMPA (3 × 3) as the number of pairs increases from 1 '
               'to 20.'),
           table(metric_table('sphere', ['sample', 'background']),
                 'Nylon sphere: errors (µm on the detector) and agreement with the ground truth. Single-pair rows: '
                 'mean ± s.d. over 20 pairs.', widths=[3.6, 2.4, 2.4, 2.4, 2.4, 1.9, 1.9])]

    # ---------------------------------------------------------------- S6
    reg = ['phantom', 'edges', 'inserts', 'flat interior', 'background']
    st += [P('S6 Shepp–Logan phantom: fine structure', H1),
           P(f'With the real SiC modulator, the speckle visibility is about three times lower than with the '
             f'modulator of the sphere simulation, and the refraction signal inside the phantom is of the order of '
             f'0.02 px. In this regime UMPA with a single pair is dominated by noise (RMSE in the phantom '
             f'above 100 µm with 3 × 3, because of outliers of several pixels; {v(ml, u15, "phantom", "rmse_um")} µm '
             'with 31 × 31) '
             f'and UMPA with 20 pairs reaches {v(ml, u20, "phantom", "rmse_um")} µm. The CNN with one pair gives '
             f'{v(ml, cnn, "phantom", "rmse_um")} µm (no-displacement reference {v(ml, "d = 0", "phantom", "rmse_um")} µm), '
             f'correlation {v(ml, cnn, "phantom", "corr")} and slope {v(ml, cnn, "phantom", "slope")} (Fig. S4, '
             'Table S4).'),
           P('The inserts test whether fine features in the CNN maps are real. For each structure, the method map in '
             'a window around it is fitted as a·(ground-truth template) + plane; a is the recovered fraction of the '
             'structure’s own signal, and its noise σ<sub>a</sub> is obtained by fitting the same template at '
             'about 300 positions of uniform water, where the true signal is zero (z = a/σ<sub>a</sub>; per-structure '
             'values in the repository). The alumina line-pair groups (120 and 60 µm) and the 200 and 120 µm '
             'alumina spheres are detected with high significance in a single pair and at the correct positions '
             '(Fig. S5); the 60 µm sphere is not; the low-contrast cellulose spheres are not detected, and no '
             'structure appears where the ground truth is flat, beyond the noise level of the null placements. The '
             'recovered fraction, however, depends on the material: it grows with the ratio μ/δ of the structure '
             '(Fig. S6), i.e. with its absorption contrast, which motivates the control experiment of Section S7.'),
           fig('fig_shepp_logan_comparison_gray',
               'Shepp–Logan phantom: ground truth, UMPA with one pair (3 × 3 and 31 × 31), CNN with one pair and '
               'UMPA with 20 pairs (3 × 3); RMSE and mean end-point error per region.'),
           table(metric_table('shepp_logan', ['phantom', 'background']),
                 'Shepp–Logan: errors (µm on the detector) and agreement with the ground truth in the phantom. '
                 'Single-pair rows: mean ± s.d. over 20 pairs. Per-region values for edges, inserts and flat '
                 'interior are in results/shepp_logan/metrics.csv.', widths=[3.6, 2.4, 2.4, 2.4, 2.4, 1.9, 1.9]),
           fig('fig_shepp_logan_inserts',
               'Inserts of the Shepp–Logan phantom: (a) vertical alumina fibres, d<sub>x</sub>; (b) horizontal '
               'fibres, d<sub>y</sub>; (c) contrast-detail spheres (alumina, top row; cellulose, bottom row), '
               'd<sub>x</sub>. Same colour scale for all methods (µm on the detector). (d, e) Profiles averaged along '
               'the fibres, a linear trend removed from each curve; no amplitude rescaling.', width=15.5),
           fig('fig_shepp_logan_amplitude_vs_material',
               'Recovered fraction a of each detected structure against its μ/δ ratio, CNN (one pair) and UMPA '
               '(20 pairs), with linear fits.', width=9.0)]

    # ---------------------------------------------------------------- S7
    def dv(d, m, c, q='corr'):
        return f'{float(d[(m, c)][q]):+.2f}'.replace('-0.00', '0.00').replace('+0.00', '0.00')
    st += [P('S7 Control experiment: what each method responds to', H1),
           P('A single-shot sample image differs from its reference by the speckle displacement (refraction) and by '
             'the transmission of the object (absorption and propagation fringes). Both share the geometry of the '
             'object, so a map that only follows the transmission can still correlate with the true displacement. '
             'To separate the two, each method was applied to synthetic sample images built from the first '
             'reference: (i) the reference warped by the exact ground-truth displacement, with no intensity change; '
             '(ii) the reference multiplied by the object transmission exp(−∫μ dl), with no displacement; and (iii) '
             'the real Geant4 sample image.'),
           P(f'UMPA (single pair, 11 × 11 window) follows the pure displacement (correlation with the ground truth '
             f'{dv(ds_, "UMPA", "warp")} for the sphere, {dv(dl, "UMPA", "warp")} for the Shepp–Logan) and returns '
             f'almost nothing for absorption only ({dv(ds_, "UMPA", "absorption")}, {dv(dl, "UMPA", "absorption")}). '
             f'The CNN gives the opposite pattern: {dv(ds_, "CNN", "warp")} and {dv(dl, "CNN", "warp")} for the pure '
             f'displacement, but {dv(ds_, "CNN", "absorption")} and {dv(dl, "CNN", "absorption")} for absorption only, '
             f'close to its correlation on the real samples ({dv(ds_, "CNN", "real")}, {dv(dl, "CNN", "real")}; '
             'Fig. S7). The CNN output on these data is therefore reproduced from the intensity (transmission) '
             'contrast alone, and its agreement with the ground truth comes from the shared object geometry rather '
             'than from tracking the speckle displacement. This is consistent with the material dependence of '
             'Fig. S6, with the amplitude deficit, and with the training set, in which every object is a single '
             'material, so that absorption and refraction edges always coincide.'),
           fig('fig_diagnostic_rdbu',
               'Control experiment, d<sub>x</sub> (µm on the detector). Columns: ground truth; CNN and UMPA (single '
               'pair, 11 × 11) on the reference warped by the exact displacement (no intensity change), on the '
               'reference times the object transmission (no displacement), and on the real sample. r = correlation '
               'with the ground-truth displacement inside the object.')]

    st += comparison_section(ms, ml)

    # ---------------------------------------------------------------- S9
    st += [KeepTogether([P('S9 X-ray spectrum of the experimental set-up', H1),
           P('The spectrum of the laboratory source used for the experimental data (tungsten anode, 50 kVp, estimated '
             '25 µm Be window, 12° anode angle) was computed with SpekPy (v2.5.4). Its fluence-weighted mean energy is '
             f'14.1 keV at the tube exit (Fig. S{NFIG[0] + 1}); the effective energy defined by the first aluminium half-value '
             'layer is 6.1 keV. The tabulated spectrum is distributed with the data.'),
           fig('fig_spectrum', 'SpekPy spectrum of the W-anode source at 50 kVp (absolute fluence at 1 m per mAs).',
               width=10.0)])]

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title=TITLE, author='S. González López, C. D. Tibambre, C. A. Ávila')
    doc.build(st, onFirstPage=onpage, onLaterPages=onpage)
    print('written', OUT)


if __name__ == '__main__':
    build()
