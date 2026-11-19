"""Provenance script: converts the Geant4 .raw images and the .npz ground truths used during the revision into the
plain-text files of data/. It is kept for traceability only; everything else in this repository reads the .txt files.

Geant4 .raw: 256 x 256 float32, little endian, written in append mode by the PEPI2 RunAction (one image per file here:
the size is checked). Pixel (row i, col j) centre: x = (j - 128) P, y = (128 - i) P, so rows grow with -y. The counts
are integers and DOUBLE-COUNTED (the run is merged with itself in RunAction.cc: every pixel = 2 x detected photons);
divide by 2 before any Poisson-noise model. The .txt files keep the raw values (format %.6e, exact for these integers).

Usage (from the repository root, on the machine that holds the original files):
    python scripts/convert_raw_to_txt.py --sl-build ~/CNN_Sphere_Sample/PEPI2_SL20/build \
        --gt-sl ~/CNN_Sphere_Sample/PEPI2/analysis/shepp_logan/gt_sl_geant4_28keV.npz \
        --gt-sphere ~/CNN_Sphere_Sample/PEPI2/analysis/sphere_M2/gt_esfera_M2_28keV.npz
"""
import argparse, glob, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMT = '%.6e'


def read_raw(f):
    a = np.fromfile(f, np.float32)
    assert a.size == 65536, f'{f}: {a.size / 65536} images in file'
    return a.reshape(256, 256)


def save_check(img, out):
    np.savetxt(out, img, fmt=FMT)
    back = np.loadtxt(out)
    assert np.array_equal(back.astype(np.float32), img.astype(np.float32)), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sl-build', required=True)
    p.add_argument('--gt-sl', required=True)
    p.add_argument('--gt-sphere', required=True)
    a = p.parse_args()
    d = os.path.join(ROOT, 'data', 'shepp_logan')
    for k in range(20):
        for t in 'RS':
            f = glob.glob(os.path.join(a.sl_build, f'SL_{t}{k:02d}_28kev_SiC125_*.raw'))
            assert len(f) == 1, f
            save_check(read_raw(f[0]), os.path.join(d, 'pairs', f'{t}{k:02d}.txt'))
    for n in ('flat', 'fantoma'):
        f = glob.glob(os.path.join(a.sl_build, f'SL_check_{n}_28kev_*.raw'))[0]
        save_check(read_raw(f), os.path.join(d, 'checks', {'flat': 'flat_no_modulator.txt',
                                                             'fantoma': 'phantom_no_modulator.txt'}[n]))
    for src, ds in ((a.gt_sl, 'shepp_logan'), (a.gt_sphere, 'sphere')):
        g = np.load(src, allow_pickle=True)
        o = os.path.join(ROOT, 'data', ds, 'ground_truth')
        for key, name in (('dcol_px', 'dx_px'), ('drow_px', 'dy_px'), ('proj_delta_mm', 'proj_delta_mm'),
                          ('mu_line', 'mu_line'), ('phase_rad', 'phase_rad')):
            np.savetxt(os.path.join(o, name + '.txt'), g[key], fmt='%.9e')
        np.savetxt(os.path.join(o, 'label.txt'), g['label'], fmt='%d')
    print('done')


if __name__ == '__main__':
    main()
