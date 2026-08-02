#!/bin/bash
cd /Users/epagogellc/parzival/boussinesq
p(){ printf "%-4s %-28s %s\n" "$1" "$2" "$3"; }
echo "=== DONE CHECK $(date '+%m-%d %H:%M') ==="
grep -q "CONTRACTS" M1_GATE_LADDER.out 2>/dev/null && M1=PASS || M1=FAIL
[ -f VERDICT_ORBIT.txt ] && grep -q "s_transient" VERDICT_ORBIT.txt 2>/dev/null && M2=PASS || M2=FAIL
[ -f VERDICT_ORBIT.txt ] && grep -qE "SETTLING|WANDERING|CYCLE" VERDICT_ORBIT.txt 2>/dev/null && M3=PASS || M3=FAIL
M4=$( [ -f sigma_grid_spread.txt ] && awk '$1>=0.15{bad=1}END{print (bad||NR==0)?"FAIL":"PASS"}' sigma_grid_spread.txt || echo FAIL )
[ -f NOVELTY.md ] && M5=PASS || M5=FAIL
p M1 "s-march correct"        $M1
p M2 "past the S6 transient"  $M2
p M3 "orbit verdict"          $M3
p M4 "sigma_Lambda magnitude" $M4
p M5 "novelty settled"        $M5
echo "certification: $(cat VERIFY_STATUS 2>/dev/null)"
if [ "$M1$M2$M3$M4$M5" = "PASSPASSPASSPASSPASS" ]; then
  echo ">>> PROJECT COMPLETE. Write it up."
else
  echo ">>> NOT DONE. Next: $( [ "$M1" = FAIL ] && echo 'M1 (fix T2+T4, HANDOFF_SMARCH.md build order)' || ([ "$M2" = FAIL ] && echo 'M2 (march past transient)' || ([ "$M3" = FAIL ] && echo 'M3 (orbit verdict)' || ([ "$M4" = FAIL ] && echo 'M4 (grid spread)' || echo 'M5 (novelty search)'))))"
fi
