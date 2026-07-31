#!/bin/bash
cd /Users/epagogellc/parzival/boussinesq
V=/Users/epagogellc/parzival/.venv-dedalus/bin/python
echo "================ $(date '+%H:%M') ================"
echo "CERTIFICATION: $(cat VERIFY_STATUS 2>/dev/null)"
echo
echo "--- VISCOUS SIGN INVERSION (cross-grid) ---"
$V sigma_peak.py 2>/dev/null | tail -7
echo
echo "--- WANDERING SCAN (generic ICs, 512x1536) ---"
for S in 11 42 77; do
  L=/tmp/G5_$S.log
  if [ ! -f $L ]; then echo "seed $S: queued"
  elif grep -q AXISYM $L; then echo "seed $S: DONE $(grep AXISYM $L | tail -1 | cut -c22-95)"
  else echo "seed $S: running $(tail -1 $L | grep -o 't=[0-9.e-]* it=[0-9]*' | head -1) of t=0.0040"
  fi
done
echo
$V orbit_speed.py G5_11 G5_42 G5_77 2>/dev/null | grep -v '^ANCHOR' || echo "(no orbit data yet)"
echo "reference: corner IC = SETTLING; noise floor 0.005"
echo
$V bottomline.py
