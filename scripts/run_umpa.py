"""UMPA (Zdora et al. 2017; package UMPA v0.2, https://github.com/optimato/UMPA) on the .txt pairs.

UMPA.match(sam, ref, Nw, df) with a Hamming window of (2 Nw + 1)^2 px; UMPA v0.2 ignores max_shift (+-4 px search).
The bias of the method is removed by subtracting UMPA(ref, ref) computed with the same pairs and window. The output is
cropped by UMPA; it is padded back to 256 x 256 with NaN, and multiplied by UMPA_SIGN = -1 to the physical sign of the
ground truth (raw UMPA returns minus the speckle displacement of common.py's convention).
Dark-field model: df = True for the sphere (setting of the manuscript notebooks), False for the Shepp-Logan (as in
the revision analysis; both give the same displacement maps within noise at these signal levels).

Configurations computed (N = number of pairs, pairs taken in file order):
    sphere      N = 1 (Nw 1, 3, 15), 5, 10, 15, 20 (Nw 1); every single pair with Nw 1 and Nw 15
    shepp_logan N = 1 (Nw 1, 3, 15), 20 (Nw 1, 3); every single pair with Nw 1 and Nw 15
Usage (python with UMPA, e.g. the system python3 here):  python3 run_umpa.py --dataset sphere [--diagnostic]
"""
import argparse, glob, os, time
import numpy as np
import UMPA
from common import GEOM, RESULTS, UMPA_SIGN, load_stack, pad_to_full, save_map

CONFIG = {'sphere': dict(df=True, multi=[(1, 1), (1, 3), (1, 15), (5, 1), (10, 1), (15, 1), (20, 1)]),
          'shepp_logan': dict(df=False, multi=[(1, 1), (1, 3), (1, 15), (20, 1), (20, 3)])}
SINGLE_NW = (1, 15)
DIAG_NW = 5


def umpa(sam, ref, nw, df):
    sam, ref = np.atleast_3d(sam.T).T.copy(), np.atleast_3d(ref.T).T.copy()      # (N, 256, 256)
    u = UMPA.match(sam, ref, nw, df=df)
    b = UMPA.match(ref, ref.copy(), nw, df=df)
    return (UMPA_SIGN * pad_to_full(u['dx'] - b['dx']), UMPA_SIGN * pad_to_full(u['dy'] - b['dy']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=list(GEOM), required=True)
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--skip-single', action='store_true', help='do not run every single pair')
    a = p.parse_args()
    cfg = CONFIG[a.dataset]
    if a.diagnostic:
        d = os.path.join(RESULTS, a.dataset, 'diagnostic')
        ref = np.loadtxt(os.path.join(d, 'inputs', 'reference.txt'))
        for f in sorted(glob.glob(os.path.join(d, 'inputs', 'sample_*.txt'))):
            case = os.path.basename(f)[7:-4]
            dx, dy = umpa(np.loadtxt(f), ref, DIAG_NW, cfg['df'])
            save_map(dx, d, f'umpa_Nw{DIAG_NW}_{case}_dx.txt'); save_map(dy, d, f'umpa_Nw{DIAG_NW}_{case}_dy.txt')
            print('diagnostic', case, flush=True)
        return
    pairs = GEOM[a.dataset]['pairs']
    R, S = load_stack(a.dataset)
    out = os.path.join(RESULTS, a.dataset, 'umpa')
    for n, nw in cfg['multi']:
        t = time.time()
        dx, dy = umpa(S[:n], R[:n], nw, cfg['df'])
        save_map(dx, out, f'N{n:02d}_Nw{nw}_dx.txt'); save_map(dy, out, f'N{n:02d}_Nw{nw}_dy.txt')
        print(f'N {n} Nw {nw}: {time.time() - t:.1f} s', flush=True)
    if a.skip_single:
        return
    for nw in SINGLE_NW:
        for i, k in enumerate(pairs):
            dx, dy = umpa(S[i], R[i], nw, cfg['df'])
            save_map(dx, out, 'single', f'pair_{k:02d}_Nw{nw}_dx.txt')
            save_map(dy, out, 'single', f'pair_{k:02d}_Nw{nw}_dy.txt')
        print(f'single pairs Nw {nw} done', flush=True)


if __name__ == '__main__':
    main()
