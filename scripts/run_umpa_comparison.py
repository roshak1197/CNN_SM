"""UMPA for the comparison of section S8 on the third-party datasets (the sphere and Shepp-Logan runs are in
run_umpa.py). Bias UMPA(ref, ref) subtracted, assign_coordinates = 'sam' (UMPA default; 'ref' flips the sign and
creates a bipolar edge artefact at sub-pixel shifts), output padded to full size with NaN and multiplied by UMPA_SIGN.
  single pair  31 x 31 window (Nw = 15, the setting of Qiao et al. 2026 for single-pair UMPA), every pair
  reference    all pairs, 7 x 7 window (Nw = 3): MIST 15, fly 20, ESPINNet data 20 (no reference for the Star: analytic GT);
               also each half of the pairs separately, to measure the reliability of the reference
Search range: max_shift 3 px (MIST, fly: sub-pixel regime) and 10 px (ESPINNet data: shifts up to ~7 px); Star:
UMPA.match (+-4 px), df False.
Output: results/comparison/<ds>/umpa31/pair_KK_{dx,dy}.txt.gz and results/comparison/<ds>/reference/umpa_all_{dx,dy}.txt.gz
Usage: python3 run_umpa_comparison.py [datasets...] [--halves-only]   (needs UMPA)
"""
import os, sys, time
import numpy as np
import UMPA
from common import RESULTS, THIRD, UMPA_SIGN, load_pair_any
from run_networks_io import save

SHIFT = {'mist': 3, 'fly': 3, 'espinnet': 10}


def full(a, shape):
    o = np.full(shape, np.nan); p0, p1 = (shape[0] - a.shape[0]) // 2, (shape[1] - a.shape[1]) // 2
    o[p0:p0 + a.shape[0], p1:p1 + a.shape[1]] = a; return o


def umpa(S, R, nw, ds):
    S, R = [np.ascontiguousarray(x, np.float64) for x in S], [np.ascontiguousarray(x, np.float64) for x in R]
    if ds == 'star':
        a, b = UMPA.match(S, R, nw, df=False), UMPA.match(R, [x.copy() for x in R], nw, df=False)
    else:
        def m(s, r):
            o = UMPA.model.UMPAModelDF(s, r, window_size=nw, max_shift=SHIFT[ds]); o.assign_coordinates = 'sam'
            return o.match(num_threads=int(os.environ.get('THREADS', 24)))
        a, b = m(S, R), m(R, [x.copy() for x in R])
    return np.array([UMPA_SIGN * full(a[c] - b[c], R[0].shape) for c in ('dx', 'dy')])


def main():
    flags = [x for x in sys.argv[1:] if x.startswith('--')]
    for ds in [x for x in sys.argv[1:] if not x.startswith('--')] or ['star', 'mist', 'fly', 'espinnet']:
        t = time.time(); RS = [load_pair_any(ds, k) for k in THIRD[ds]['pairs']]
        if ds != 'star' and '--halves-only' not in flags:
            save(umpa([s for _, s in RS], [r for r, _ in RS], 3, ds), ds, 'reference', 'umpa_all')
            print(ds, 'reference', f'{time.time() - t:.0f} s', flush=True)
        if ds != 'star':
            h = len(RS) // 2                                     # reliability of the reference: two independent halves
            for name, part in (('umpa_half1', RS[:h]), ('umpa_half2', RS[h:])):
                save(umpa([s for _, s in part], [r for r, _ in part], 3, ds), ds, 'reference', name)
        if '--halves-only' in flags:
            continue
        for k, (r, s) in zip(THIRD[ds]['pairs'], RS):
            save(umpa([s], [r], 15, ds), ds, 'umpa31', f'pair_{k:02d}')
        print(ds, 'single pairs', f'{time.time() - t:.0f} s', flush=True)


if __name__ == '__main__':
    main()
