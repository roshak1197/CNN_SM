#!/bin/bash
# Reproduce every result, figure and the supplementary PDF from the .txt data.
# PY_CNN: python with torch (CNN inference). PY_UMPA: python with UMPA. Both need numpy, scipy, matplotlib.
# Defaults are the interpreters used on the original server; override, e.g.  PY_CNN=python PY_UMPA=python bash run_all.sh
set -e
cd "$(dirname "$0")/scripts"
PY_CNN=${PY_CNN:-python}
PY_UMPA=${PY_UMPA:-python}
for ds in sphere shepp_logan; do
  $PY_CNN  make_diagnostic_inputs.py --dataset $ds
  $PY_CNN  run_cnn.py  --dataset $ds
  $PY_CNN  run_cnn.py  --dataset $ds --diagnostic
  $PY_UMPA run_umpa.py --dataset $ds
  $PY_UMPA run_umpa.py --dataset $ds --diagnostic
done
$PY_UMPA gt_consistency_check.py
$PY_CNN metrics.py
$PY_CNN inserts_analysis.py
$PY_CNN figures.py
$PY_UMPA make_supplementary.py      # needs reportlab
