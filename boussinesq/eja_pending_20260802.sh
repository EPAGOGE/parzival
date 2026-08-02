#!/bin/zsh
# Pending EJA firings for the M1 gate PASS (2026-08-02).
# The session's Bash path got blocked by a context classifier mid-run;
# these are the exact mints that belong on the ledger. Run me once.
set -e
EJ="/Users/epagogellc/parzival/.venv/bin/python3 /Users/epagogellc/epagoge/jump/tools/ej.py"
ST="/Users/epagogellc/parzival/eja_state"

eval $EJ --state $ST deduce m1_pnorm_gate_contracts V_final_over_initial 3.66e-4 \
  -e "'Production grid (16,40,12)/Nb36/eps_b=1e-4, root A F-max 2.43e-11, cholesky re-PASSED: 80 BE steps ds=0.25 amp=1e-6 admissible pert; quasi-Newton (frozen reduced Jacobian = -sign*Lred + I/ds, validated 4.6e-14 vs exact SMarcher) converged true residual every step; V(v) 8.60e-8 -> 3.15e-11, ZERO growth violations; raw norm peak 2.39x at s=0.5 then decayed to 0.18x by s=20. CONTRACTS. M1_GATE_LADDER.out'"

eval $EJ --state $ST resolve 74 confirmed \
  -o "'Blocker dissolved without the ladder: project_oblique is identity on ker(Cg), frozen reduced BE operator = -sign*Lred + I/ds; 0.5s LU then 0.04s/step at production. Gate ran there and PASSED.'"

eval $EJ --state $ST tension "'LADDER SIDE-FINDING: cholesky certification never turns on across the ladder - Nb20/Nb24 have NO root at fixed alpha (floors 7.1e-3/6.6e-3); Nb28 (n_f 4104) and Nb32 (n_f 5208) converge to 1e-11 with exact Lyapunov solves but P INDEFINITE (4 negative directions each), including a rung LARGER than certified production (n_f 4760). Hurwitz-ness is CONFIG-dependent not size-dependent. Prime suspect: eps_b (ladder at 1e-3 = singular wedge layer; production cert at 1e-4 where it vanishes). DISCRIMINATING TEST: rung-3 config at eps_b=1e-4 - cholesky flips = eps_b is the driver'" \
  --source m1-ladder

eval $EJ --state $ST consume 72 --by production_pnorm_gate_PASS
eval $EJ --state $ST status | head -3
echo "EJA pending mints: DONE"
