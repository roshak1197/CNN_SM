#!/bin/bash
# Compile (if needed) and run the Shepp-Logan images one after the other (one Geant4 process per image:
# phantom on/off and modulator position are fixed at /run/initialize).
# Usage:  bash correr.sh [threads] [macro ...]      e.g.  bash correr.sh 60   or   bash correr.sh 30 run_S00.mac
# Default macros: run_check_flat.mac run_check_fantoma.mac run_R00.mac run_S00.mac run_R01.mac run_S01.mac run_R02.mac run_S02.mac run_R03.mac run_S03.mac run_R04.mac run_S04.mac run_R05.mac run_S05.mac run_R06.mac run_S06.mac run_R07.mac run_S07.mac run_R08.mac run_S08.mac run_R09.mac run_S09.mac run_R10.mac run_S10.mac run_R11.mac run_S11.mac run_R12.mac run_S12.mac run_R13.mac run_S13.mac run_R14.mac run_S14.mac run_R15.mac run_S15.mac run_R16.mac run_S16.mac run_R17.mac run_S17.mac run_R18.mac run_S18.mac run_R19.mac run_S19.mac
set -e
cd "$(dirname "$0")"
H=${1:-30}; shift || true
M=("$@"); [ ${#M[@]} -eq 0 ] && M=(run_check_flat.mac run_check_fantoma.mac run_R00.mac run_S00.mac run_R01.mac run_S01.mac run_R02.mac run_S02.mac run_R03.mac run_S03.mac run_R04.mac run_S04.mac run_R05.mac run_S05.mac run_R06.mac run_S06.mac run_R07.mac run_S07.mac run_R08.mac run_S08.mac run_R09.mac run_S09.mac run_R10.mac run_S10.mac run_R11.mac run_S11.mac run_R12.mac run_S12.mac run_R13.mac run_S13.mac run_R14.mac run_S14.mac run_R15.mac run_S15.mac run_R16.mac run_S16.mac run_R17.mac run_S17.mac run_R18.mac run_S18.mac run_R19.mac run_S19.mac)
mkdir -p build && cd build
[ -x PEPI2 ] || { cmake .. > compilacion.log 2>&1 && make -j"$(nproc)" >> compilacion.log 2>&1; }
cmake .. > /dev/null                                   # refresh the macros copied into build/
for m in "${M[@]}"; do
  base=$(grep -oP '(?<=setBaseName )\S+' "$m")
  if ls "${base}"_d*.raw > /dev/null 2>&1; then echo "skip $m (${base}*.raw exists; .raw files are appended)"; continue; fi
  sed -i "s|^/run/numberOfThreads .*|/run/numberOfThreads $H|" "$m"
  echo "$(date '+%F %T') start $m ($H threads)"
  ./PEPI2 "$m" > "${m%.mac}.log" 2>&1
  grep -q "SL_CONFIG" "${m%.mac}.log" || { echo "ERROR: no SL_CONFIG line in ${m%.mac}.log"; exit 1; }
  grep -i "overlap" "${m%.mac}.log" | grep -vi "OK" | head -5 || true
  echo "$(date '+%F %T') done  $m -> $(ls ${base}_d*.raw)"
done
