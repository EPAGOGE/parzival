# M2 AMPLITUDE-LADDER SPEC (staged 2026-08-02 evening)

The next mini test on the solve, ready to fire. M1 PASSED at production
(P-norm gate, M1_GATE_LADDER.out); this is the first measurement PAST the
transient in rescaled time. New-chat opening line:

> Read parzival/boussinesq/HANDOFF_2026-08-02_EVENING.md, then
> M2_AMPLADDER_SPEC.md, then run the experiment exactly as specified.
> Terse; tool output to files, not chat.

## The experiment

Four marches at the PRODUCTION grid (root A via pnorm_P.npz certificate;
re-verify cholesky(P) on load; reuse march_s.py's QuasiNewtonSMarcher /
run_production_pnorm_gate machinery, additive edits only):

- amps: 1e-6 (control), 1e-5, 1e-4, 1e-3
- each: admissible perturbation (_admissible_pert, seed 0), sign +1.0,
  ds=0.25, 240 steps (60 s-units)
- per step: ||v||_P = sqrt(v^T P v), raw relative norm, quasi-Newton
  iterations + final residual
- validity: the TRUE nonlinear residual must converge each step (M1-gate
  tolerance). QN > ~15 iters or residual stall => STOP that amp cleanly and
  record step/s/state norms — the stall point is DATA (nonlinear escape).

## Analysis (write M2_AMPLADDER.out)

1. P-norm monotonicity per amp. ANY P-norm growth at finite amplitude is a
   headline: linear flow cannot grow in this norm; growth = genuinely
   nonlinear effect (the wandering sector's first direct signature).
2. s_transient per amp: where the raw norm, after its peak, re-crosses BELOW
   initial (DONE.md M2 criterion). Report s_peak and peak size too.
3. Escape threshold: largest amp completing 60 s-units; where escape occurs.
4. Linear-regime check: raw trajectories divided by amp should collapse;
   report where collapse breaks.

If any amp >= 1e-5 genuinely meets M2's criterion with the march valid
throughout: write VERDICT_ORBIT_DRAFT.txt with s_transient per amp (DRAFT
deliberately — flipping done.sh milestones is the operator's call).

## Refusals (binding, as always)

Definiteness by cholesky only; no eigenvalue readouts for decisions; no
hand-rolled projectors; additive march_s.py edits only; new output files
only; runs/stream_*.jsonl untouched; laptop only; honest wall times.

Python: /Users/epagogellc/parzival/.venv/bin/python
