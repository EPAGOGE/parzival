#!/bin/bash
# Completion daemon: waits for all 8 seeds, runs the gated analysis,
# writes VERDICT.txt. No interaction.
cd /Users/epagogellc/parzival/boussinesq
V=/Users/epagogellc/parzival/.venv-dedalus/bin/python
SEEDS="3 11 19 23 42 51 63 77"
for i in $(seq 1 180); do
  DONE=0
  for S in $SEEDS; do grep -q AXISYM /tmp/W$S.log 2>/dev/null && DONE=$((DONE+1)); done
  echo "$(date '+%H:%M') $DONE/8 complete" >> finish.log
  [ "$DONE" -ge 8 ] && break
  sleep 60
done
{
  echo "GENERIC-IC WANDERING SCAN -- FINAL"
  echo "$(date '+%Y-%m-%d %H:%M')  |  8 seeds, 512x1536, gated (tail<=1e-6, gamma<=1e-4)"
  echo
  $V orbit_speed.py $(for S in $SEEDS; do echo -n "W$S "; done)
  echo
  $V - <<'PY'
import subprocess,re,sys,numpy as np
seeds=[3,11,19,23,42,51,63,77]
out=subprocess.run([sys.executable,"orbit_speed.py"]+[f"W{s}" for s in seeds],
                   capture_output=True,text=True).stdout
rows=[]
for ln in out.splitlines():
    m=re.match(r"(W\d+)\s+slope\s+([-+\d.]+)\s+90% CI \[([-+\d.]+),([-+\d.]+)\].*?-> (\S+)",ln)
    if m: rows.append((m.group(1),float(m.group(2)),float(m.group(3)),float(m.group(4)),m.group(5)))
print("="*70); print("VERDICT")
if not rows:
    print("  NO SEED READABLE. The 512 grid does not resolve generic data long")
    print("  enough to measure an orbit. That is itself the finding: the")
    print("  wandering sector is inaccessible at laptop resolution."); print("="*70); sys.exit()
labs=[r[4] for r in rows]; sl=np.array([r[1] for r in rows])
n_set=labs.count("SETTLING"); n_wan=sum(1 for l in labs if l.startswith("WANDERING"))
print(f"  {len(rows)}/8 seeds readable.  SETTLING {n_set} | WANDERING {n_wan} | other {len(rows)-n_set-n_wan}")
print(f"  slope mean {sl.mean():+.3f}  sd {sl.std():+.3f}  range [{sl.min():+.3f},{sl.max():+.3f}]")
if n_set==len(rows):
    print("  ALL SEEDS SETTLE. Generic near-wall data converges onto a profile.")
    print("  No wandering at this depth. Scope: ~1.3 e-foldings of growth;")
    print("  silent about 10+ e-foldings where wandering would live.")
elif n_wan==len(rows):
    print("  ALL SEEDS WANDER. Generic orbits never settle: first measured")
    print("  footprint in the sector with no analytic tools.")
else:
    print("  SEEDS DISAGREE -> genericity itself is the finding: identical")
    print("  equations, different data, different fate.")
print("="*70)
PY
} > VERDICT.txt 2>&1
cat VERDICT.txt
