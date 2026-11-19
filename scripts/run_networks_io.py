"""Writer shared by run_networks.py and run_umpa_comparison.py (no torch / UMPA import)."""
import os
import numpy as np
from common import RESULTS


def save(a, ds, method, name):
    d = os.path.join(RESULTS, 'comparison', ds, method); os.makedirs(d, exist_ok=True)
    np.savetxt(os.path.join(d, name + '_dx.txt.gz'), a[0], fmt='%.6e')
    np.savetxt(os.path.join(d, name + '_dy.txt.gz'), a[1], fmt='%.6e')
