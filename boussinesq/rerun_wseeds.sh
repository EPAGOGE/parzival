#!/bin/zsh
# CURIOUS W2 kill test, decisive arm: regenerate three generic W-seed runs
# (full-field snapshots WERE deleted; streams survive) under NEW run-ids so
# the original stream_W*.jsonl files are never truncated. Then rerun the
# |D|-bars bootstrap over everything. Laptop-only.
cd /Users/epagogellc/parzival/boussinesq
PY=/Users/epagogellc/parzival/.venv-dedalus/bin/python
for S in 3 63 77; do
  nohup $PY dedalus_axisym.py --scenario --Nz 256 --Nr 768 --ic-power 4 \
    --ic-generic $S --tmax 0.0013 --ckpt-sim-dt 2.5e-5 \
    --run-id W${S}R --out ../runs/ax_W${S}R.json > wseed_W${S}R.log 2>&1 &
done
wait
$PY curious_dbars.py OR_z256r768 OR_z256r384 OR_z128r768 OR_z128r384 \
  G5_11 NUL1e-3 NUL1e-4 N2_1e-3 N2_1e-4 SYMa SYMb loc1024 mpi1024 \
  W3R W63R W77R > /dev/null 2>> dbars_err.log
echo DONE > WSEED_RERUN.done
