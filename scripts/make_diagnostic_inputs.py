"""Synthetic single-shot inputs that separate what a method responds to (control experiment).

From the reference R of the first pair of the dataset:
  warp          R(r - d_GT): the exact ground-truth speckle displacement, no intensity change (pure refraction)
  absorption    R * exp(-int mu dl): the transmission of the object, NO speckle displacement
  real          the real Geant4 sample image of the same modulator position (everything)
A method that tracks the speckle displacement must reproduce d_GT for 'warp' and give ~0 for 'absorption'.
Writes results/<dataset>/diagnostic/inputs/{reference,sample_warp,sample_absorption,sample_real}.txt.
Then:  python run_cnn.py --dataset X --diagnostic ;  python3 run_umpa.py --dataset X --diagnostic
Usage: python make_diagnostic_inputs.py --dataset sphere
"""
import argparse, os
import numpy as np
from scipy.ndimage import map_coordinates
from common import GEOM, RESULTS, load_gt, load_pair, save_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(GEOM), required=True)
    a = p.parse_args()
    k = GEOM[a.dataset]['pairs'][0]
    R, S = load_pair(a.dataset, k)
    g = load_gt(a.dataset)
    yy, xx = np.mgrid[0:256, 0:256].astype(float)
    cases = {'warp': map_coordinates(R, [yy - g['dy_px'], xx - g['dx_px']], order=3, mode='reflect'),
             'absorption': R * np.exp(-g['mu_line']), 'real': S}
    d = os.path.join(RESULTS, a.dataset, 'diagnostic', 'inputs')
    save_map(R, d, 'reference.txt')
    for n, img in cases.items():
        save_map(img, d, f'sample_{n}.txt')
    print('pair', k, 'written to', d)


if __name__ == '__main__':
    main()
