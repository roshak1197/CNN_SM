# CNN_SM — data, code and supplementary material

Companion repository of the manuscript **"Single-shot bi-directional phase retrieval in X-ray speckle-based imaging:
a convolutional neural network approach"** (S. González López, C. D. Tibambre, C. A. Ávila; J. Phys. D, under review).

It contains the authors' own simulated data (Geant4), the analytic ground truths, the CNN and its weights, the
scripts that produce every number and figure of the supplementary material, and the supplementary PDF itself.
The comparison with other methods (UMPA, StrainNet, ESPINNet) on our phantoms and on four third-party datasets is
part of the pipeline (section S8 of the PDF), but third-party code, weights and data are **not** redistributed:
`third_party/fetch_third_party.sh` downloads them from their original sources.

## Layout

```
data/
  sphere/        pairs/R01..R20.txt, S01..S20.txt        Nylon sphere, 20 modulator positions
                 ground_truth/*.txt                      analytic GT (see below)
                 cnn_manuscript_d[xy].txt                CNN map shown in the manuscript (pair 1; see notes)
  shepp_logan/   pairs/R00..R19.txt, S00..S19.txt        Shepp-Logan phantom + inserts, 20 positions
                 checks/flat_no_modulator.txt, phantom_no_modulator.txt   transmission check runs (2e8 photons)
                 ground_truth/*.txt
  spectrum/      spectrum_W50kVp_Be25um_spekpy.txt       SpekPy spectrum of the laboratory source
cnn/             FlowNet.py (FlowNetPyramidal), checkpoints/CNN.pt (pre-trained), finetuned_best_model.pt
geant4/          shepp_logan/  Geant4 application that produced data/shepp_logan (sources, macros, logs)
                 preparar_geant4_sl.py  generator of that application from the sphere project (provenance)
third_party/     fetch_third_party.sh (downloads ESPINNet, StrainNet, MIST, fly; nothing committed), README.md
scripts/         analysis pipeline (reads only .txt)
results/         metric tables (maps are regenerated, see .gitignore)
figures/         every figure, PNG + PDF, diverging (rdbu) and gray versions
supplementary/   Supplementary_Material.pdf
```

## Data

All images are 256 × 256 plain-text matrices (`numpy.loadtxt`), in the orientation of the Geant4 detector: pixel
(row i, column j) is centred at x = (j − 128)·P, y = (128 − i)·P, P = 55 µm (rows grow with −y).

| | Nylon sphere | Shepp–Logan phantom |
|---|---|---|
| object | sphere, R = 2 mm (4 mm diameter), C6H11NO 1.4 g/cm³ | 10 nested ellipsoids (scale 4 mm) + 19 inserts (alumina fibres Ø 120/60 µm, graphite box, alumina/cellulose spheres Ø 200/120/60 µm) |
| source–object / object–detector | 75 cm / 75 cm (M = 2) | 85 cm / 65 cm (M = 1.765) |
| energy, photons | 28 keV mono, 1e9 primaries per image | same |
| modulator | 125 µm grains, 4 layers | real SiC (3.21 g/cm³) 125 µm grains, 4 layers on 200 µm cellulose |
| pairs | R01–R20 / S01–S20 | R00–R19 / S00–S19 (Geant4 run numbers; offsets in Table S2 of the PDF) |

* R_k = reference (no sample), S_k = sample, same modulator position k.
* Values are detector counts **as recorded**: the Geant4 application counts every photon twice (the run is merged with
  itself in `RunAction.cc`), so Poisson statistics correspond to half of the stored value. The CNN normalises each
  pair and UMPA is insensitive to a global factor; anything that models Poisson noise must divide by 2.
* The beam is a cone: the detector corners (r > 164 px) are dark and excluded from every metric.
* The Shepp–Logan `.txt` files were converted from the Geant4 `.raw` (float32) by `scripts/convert_raw_to_txt.py`; the
  conversion is exact (checked bit by bit). The sphere pairs were already provided as text.

**Ground truth** (`ground_truth/`, from `scripts/gt_sphere.py`, `scripts/gt_shepp_logan.py`): exact ray tracing through
the object in the cone-beam geometry, 4 × 4 rays per pixel.
`dx_px`, `dy_px` = speckle displacement along +column / +row in detector pixels, **physical sign**
d = −(z2/P) ∂(∫δ dl)/∂x_obj; `proj_delta_mm` = ∫δ dl; `mu_line` = ∫μ dl; `phase_rad` = −(2π/λ)∫δ dl;
`label` = 0 background, 1 object (sphere) or 1–10 ellipsoid regions and 11 + k insert k (Shepp–Logan).
δ values are those of the Geant4 refraction tables (data/*_delta.txt of the Geant4 application).

## Third-party material (comparison, section S8)

| What | Source | Licence | How |
|---|---|---|---|
| ESPINNet code + weights | Qiao et al., J. Synchrotron Rad. 33, 467 (2026), github.com/QDLDQQiao/ESPINNet | MIT | `git clone` |
| ESPINNet test data | OneDrive link in the ESPINNet README | (with the repository) | manual download |
| StrainNet code + Star frames | Boukhtache et al. 2021, github.com/DreamIP/StrainNet | CC BY-NC-SA 4.0 | `git clone` |
| StrainNet-h weights | Google Drive link in the StrainNet README | CC BY-NC-SA 4.0 | manual download |
| MIST wattle flower data | Alloo et al., Sci. Rep. (2023), github.com/samanthaalloo/MIST_Alloo23 | MIT | `git clone` |
| Fly data | OSF project 4ruq2, osf.io/4ruq2 | not stated | OSF zip download |

`bash third_party/fetch_third_party.sh` (then the two manual downloads it lists) and
`python scripts/prepare_third_party.py`, which writes the pairs as .txt to `data/third_party/` (~0.9 GB, git-ignored)
with the preprocessing of the analysis (MIST dark/flat correction as in the authors' script; 1024 × 1024 ROIs).

## Reproducing everything

Two Python environments were used on the original server (any single environment with all packages works):
torch for the CNN, and [UMPA](https://github.com/optimato/UMPA) v0.2 for UMPA. See `environment.yml`.

```bash
PY_CNN=python PY_UMPA=python bash run_all.sh
```

Steps (all in `scripts/`): `make_diagnostic_inputs.py` → `run_cnn.py` → `run_umpa.py` → `gt_consistency_check.py` →
`prepare_third_party.py` → `run_umpa_comparison.py` → `run_networks.py` → `metrics.py` → `inserts_analysis.py` →
`figures.py` → `comparison.py` → `make_supplementary.py`. The comparison needs a few hours on CPU (1024 × 1024 images;
maps of these runs are written as .txt.gz). CNN inference runs on CPU
(≈ 4 s per pair with 8 threads). `common.py` holds the conventions (paths, geometry, masks, regions, metrics).

Conventions of every result file: displacements in detector pixels with the physical sign of the GT (raw UMPA output is
multiplied by −1); figures show micrometres on the detector (d × 55 µm), refraction angle α = d·P/z2.

## Notes

* `data/sphere/cnn_manuscript_d[xy].txt` is the CNN map used for the sphere figure of the manuscript. `CNN.pt`
  reproduces it closely (correlation 0.88, same metrics) but not exactly; all results here use `CNN.pt`.
* `finetuned_best_model.pt` (fine-tuned on 50 kVp polychromatic data) is included for completeness; it is not suited to
  these monochromatic simulations and is not used in the supplementary material.
* The Geant4 application is derived from Geant4PEPI (L. Brombal, INFN Trieste) with changes by C. Sevilla and
  C. D. Tibambre; Geant4 licence terms apply. Rebuild with Geant4 11 (`geant4/shepp_logan/correr.sh`).
* The experimental data of the manuscript (beetle) are not part of this repository.

## Licence and citation

To be defined by the authors before the repository is made public.
