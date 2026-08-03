#!/bin/bash
# NU-CURVE campaign runner (NU_CURVE_SPEC.md). Sequential, both grids,
# 5 viscosities, dense cadence. ~40 min total. New tags only.
set -e
cd "$(dirname "$0")"
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib:$(brew --prefix open-mpi)/lib:$(brew --prefix hdf5)/lib"
V=/Users/epagogellc/parzival/.venv-dedalus/bin/python
R=../runs
for nu in 1e-3 1e-4 3e-4 3e-5 1e-5; do
  for g in "128 384 C128" "256 768 C256"; do
    set -- $g
    tag="${3}_${nu}"
    if [ -d "$R/snap_$tag" ]; then echo "$tag present, skipping"; continue; fi
    echo "[$(date '+%H:%M:%S')] $tag starting"
    $V dedalus_axisym.py --scenario --Nz $1 --Nr $2 --A 100 \
       --ic-power 4 --zpow 1 --r0 0.4 --tmax 0.0030 --nu $nu \
       --ckpt-sim-dt 3e-5 --ckpt-max-writes 60 \
       --run-id "$tag" --out "$R/$tag.json" > "/tmp/nucurve_$tag.log" 2>&1
    echo "[$(date '+%H:%M:%S')] $tag done"
  done
done
echo "NU-CURVE RUNS COMPLETE"
