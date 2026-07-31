#!/bin/bash
# Regenerate every dataset verify.py checks. ~12 minutes on a laptop.
#
# The verification data is NOT committed (it is ~500 MB of HDF5). It is
# regenerated instead, which is stronger: you are not trusting our bytes, you
# are producing your own and checking that they satisfy identities forced by
# the equations before any measurement exists.
set -e
cd "$(dirname "$0")"
V="${PYTHON:-python}"
R=../runs
mkdir -p "$R"

run () {   # name, Nz, Nr, amplitude, tmax, extra
  local tag=$1 nz=$2 nr=$3 amp=$4 tmax=$5; shift 5
  if [ -d "$R/snap_$tag" ] || [ -f "$R/stream_$tag.jsonl" ]; then
    echo "  $tag  already present, skipping"; return
  fi
  echo "  $tag  (${nz}x${nr}, A=$amp, tmax=$tmax)"
  $V dedalus_axisym.py --scenario --Nz "$nz" --Nr "$nr" --A "$amp" \
     --ic-power 4 --zpow 1 --r0 0.4 --tmax "$tmax" \
     --run-id "$tag" --out "$R/$tag.json" "$@" > "/tmp/boot_$tag.log" 2>&1
}

echo "[1/3] exact amplitude-symmetry pair (the zero-free-parameter anchor)"
run SYMa 256 768 100 0.0030 --ckpt-sim-dt 6e-5 --ckpt-max-writes 60
run SYMb 256 768 200 0.0015 --ckpt-sim-dt 3e-5 --ckpt-max-writes 60

echo "[2/3] off-ray 2x2 grid factorial (Nz and Nr varied independently)"
run OR_z128r384 128 384 100 0.0030 --ckpt-sim-dt 6e-5 --ckpt-max-writes 60
run OR_z128r768 128 768 100 0.0030 --ckpt-sim-dt 6e-5 --ckpt-max-writes 60
run OR_z256r384 256 384 100 0.0030 --ckpt-sim-dt 6e-5 --ckpt-max-writes 60
run OR_z256r768 256 768 100 0.0030 --ckpt-sim-dt 6e-5 --ckpt-max-writes 60

echo "[3/3] transported-invariant check"
run F10CHK 128 384 100 0.0025

echo
echo "done. now run:  $V verify.py   (expect 15/15)"
