"""Converts the third-party datasets of the comparison (section S8) into .txt pairs, exactly with the preprocessing
used in the analysis. Input: third_party/ (fetch_third_party.sh). Output: data/third_party/<name>/pairs/R??.txt, S??.txt
(not committed; ~1.5 GB). Pairs are numbered from 00 in the order of the original files.
  mist      Alloo et al. 2023, wattle flower, 15 positions: header crop and (I - dark_k)/(FF - dark_k) as in their
            Allooetal23_SciRep_MIST.py, then ROI rows 588:1612, cols 688:1712 of the cropped image (1024 x 1024).
  fly       OSF 4ruq2, 20 pairs ref/ref_*speckKK.tif, sample/mouche_*speckKK.tif, raw counts (no dark/flat provided),
            ROI rows 450:1474, cols 300:1324.
  espinnet  ESPINNet test data, refs/ and sample_in/RandomPlate_0KK.tif, central 1024 x 1024 ROI.
  star      StrainNet Star frames: pair 00 = Noiseless (Star1 reference, Star2 deformed), pair 01 = Noisy.
Usage: python prepare_third_party.py [mist fly espinnet star]
"""
import os, re, sys
import numpy as np
import tifffile
from common import DATA, THIRD_DIR

M = 1024


def out(ds):
    d = os.path.join(DATA, 'third_party', ds, 'pairs'); os.makedirs(d, exist_ok=True); return d


def write(d, k, R, S, fmt):
    np.savetxt(os.path.join(d, f'R{k:02d}.txt'), R, fmt=fmt)
    np.savetxt(os.path.join(d, f'S{k:02d}.txt'), S, fmt=fmt)


def centre(a):
    return a[a.shape[0] // 2 - M // 2:a.shape[0] // 2 + M // 2, a.shape[1] // 2 - M // 2:a.shape[1] // 2 + M // 2]


def mist():
    base = os.path.join(THIRD_DIR, 'MIST_Alloo23')
    D = os.path.join(base, 'DATA_WattleFlower_25keV_2m_SBPCXI')
    src = open(os.path.join(base, 'Allooetal23_SciRep_MIST.py')).read()
    v = {k: int(re.search(rf'^{k}\s*=\s*(\d+)', src, re.M).group(1)) for k in ('xleft', 'xright', 'ytop', 'ybot')}
    f = lambda n: tifffile.imread(os.path.join(D, n)).astype(np.float64)[v['ytop']:v['ybot'], v['xleft']:v['xright']]
    ff = f('FF_2m.tif'); d = out('mist')
    for k in range(15):
        dc = f(f'DarkCurrent_Y{k:02d}.tif')
        R = ((f(f'ReferenceSpeckle_Y{k:02d}.tif') - dc) / (ff - dc)).astype(np.float32)[588:1612, 688:1712]
        S = ((f(f'SAMPLE_Y{k:02d}_T0.tif') - dc) / (ff - dc)).astype(np.float32)[588:1612, 688:1712]
        write(d, k, R, S, '%.8e')


def fly():
    D = os.path.join(THIRD_DIR, 'fly_4ruq2'); d = out('fly')
    ld = lambda p, k: tifffile.imread(os.path.join(D, f'{p}_40kV_5.2um_30s_12cm_53cm_speck{k:02d}.tif'))[450:1474, 300:1324]
    for k in range(1, 21):
        write(d, k - 1, ld('ref/ref', k), ld('sample/mouche', k), '%d')


def espinnet():
    D = os.path.join(THIRD_DIR, 'ESPINNet'); d = out('espinnet')
    for k in range(1, 21):
        write(d, k - 1, centre(tifffile.imread(os.path.join(D, 'refs', f'RandomPlate_{k:03d}.tif'))),
              centre(tifffile.imread(os.path.join(D, 'sample_in', f'RandomPlate_{k:03d}.tif'))), '%d')


def star():
    D = os.path.join(THIRD_DIR, 'StrainNet', 'Star_frames'); d = out('star')
    for k, c in enumerate(('Noiseless', 'Noisy')):
        write(d, k, tifffile.imread(os.path.join(D, f'{c}_frames', 'Star1.tif')),
              tifffile.imread(os.path.join(D, f'{c}_frames', 'Star2.tif')), '%d')


if __name__ == '__main__':
    for name in (sys.argv[1:] or ['mist', 'fly', 'espinnet', 'star']):
        globals()[name](); print(name, 'done', flush=True)
