#!/bin/bash
# Run the Shepp-Logan images IN PARALLEL: several Geant4 processes at once, each with at most H threads.
# Each image runs in its own folder w_<macro>/ (next to build/, so ../data still resolves): no shared PEPI.root, and
# a killed run never leaves a partial .raw in build/. A finished image (.raw of 262144 bytes) is moved to build/.
# Usage:  bash paralelo.sh [threads per image = 10] [simultaneous images = auto] [macro ...]
#   auto = (cores - current load - 2) / threads
#   e.g.  bash paralelo.sh            bash paralelo.sh 10 4            bash paralelo.sh 10 2 run_R05.mac run_S05.mac
set -u
cd "$(dirname "$0")"; export RAIZ=$(pwd)
export H=${1:-10}; P=${2:-auto}; shift $(( $# < 2 ? $# : 2 ))
if [ "$P" = auto ]; then
  carga=$(awk '{printf "%d", $1 + 0.5}' /proc/loadavg)
  P=$(( ($(nproc) - carga - 2) / H )); [ "$P" -lt 1 ] && P=1
fi
M=("$@")
if [ ${#M[@]} -eq 0 ]; then for k in $(seq -w 2 19); do M+=(run_R$k.mac run_S$k.mac); done; fi

mkdir -p build && cd build
[ -x PEPI2 ] || { echo "compiling..."; cmake .. > compilacion.log 2>&1 && make -j"$(nproc)" >> compilacion.log 2>&1 || { echo "ERROR compiling, see build/compilacion.log"; exit 1; }; }
cmake .. > /dev/null
for f in SL_*.raw; do
  [ -e "$f" ] || continue
  t=$(stat -c %s "$f")
  if [ $((t % 262144)) -ne 0 ] || [ "$t" -eq 0 ]; then
    echo "ERROR: build/$f has $t bytes (a good image is 262144). Delete it and relaunch."; exit 1; fi
done
cd ..

pend=()
for m in "${M[@]}"; do
  base=$(grep -oP '(?<=setBaseName )\S+' "build/$m")
  if ls build/"${base}"_d*.raw > /dev/null 2>&1; then echo "skip $m (already in build/)"; else pend+=("$m"); fi
done
echo "$(date '+%F %T') ${#pend[@]} images pending, $P at a time x $H threads"
[ ${#pend[@]} -eq 0 ] && exit 0

una() {
  m=$1; w="$RAIZ/w_${m%.mac}"; log="$RAIZ/build/${m%.mac}.log"
  rm -rf "$w" && mkdir "$w" && cd "$w" || exit 1
  ln -s ../build/PEPI2 PEPI2
  cp ../build/detmask_config.in ../build/geometry_config.in ../build/gps.in ../build/"$m" .
  sed -i "s|^/run/numberOfThreads .*|/run/numberOfThreads $H|" "$m"
  echo "$(date '+%F %T') start $m"
  ./PEPI2 "$m" > "$log" 2>&1
  raw=$(ls SL_*.raw 2>/dev/null | head -1)
  if grep -q SL_CONFIG "$log" && [ -n "$raw" ] && [ "$(stat -c %s "$raw")" -eq 262144 ]; then
    mv "$raw" ../build/ && cd .. && rm -rf "$w"
    echo "$(date '+%F %T') done  $m -> $raw"
  else
    echo "$(date '+%F %T') FAILED $m (see $log; folder $w kept)"
  fi
}
export -f una
printf '%s\n' "${pend[@]}" | xargs -P "$P" -I{} bash -c 'una {}'
echo "$(date '+%F %T') end; images in build/: $(ls build/SL_[RS]*.raw | wc -l)"
