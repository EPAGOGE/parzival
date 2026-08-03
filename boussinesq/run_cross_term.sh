#!/bin/bash
set -e
cd "$(dirname "$0")"
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib:$(brew --prefix open-mpi)/lib:$(brew --prefix hdf5)/lib"
V=/Users/epagogellc/parzival/.venv-dedalus/bin/python
R=../runs
run_one () { # tag Nz Nr A tmax dt
  if [ -d "$R/snap_$1" ]; then echo "$1 present, skipping"; return; fi
  echo "[$(date '+%H:%M:%S')] $1 starting"
  $V dedalus_axisym.py --scenario --Nz $2 --Nr $3 --A $4 \
     --ic-power 4 --zpow 1 --r0 0.4 --tmax $5 --nu 1e-3 \
     --ckpt-sim-dt $6 --ckpt-max-writes 60 \
     --run-id "$1" --out "$R/$1.json" > "/tmp/xterm_$1.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $1 done"
}
run_one B128_A50n3  128 384  50 0.0060 6e-5
run_one B256_A50n3  256 768  50 0.0060 6e-5
run_one B128_A200n3 128 384 200 0.0015 1.5e-5
run_one B256_A200n3 256 768 200 1.5e-3 1.5e-5
echo "CROSS-TERM RUNS COMPLETE"
