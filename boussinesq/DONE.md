# DONE: the exit conditions

There was no definition of done. That is why this could run forever. Here it
is, falsifiable, checkable by `./done.sh`.

## The project

Decide, for the near-wall axisymmetric class, whether the orbit in profile
space SETTLES or WANDERS -- and report sigma_Lambda's fate under viscosity.
Not "solve Navier-Stokes". Those two questions are what the instruments can
actually reach.

## Milestones (each is PASS/FAIL, no judgement)

M1  s-march correct
    - T2 fixed: pin rows no longer reference a stale seed
    - T4 fixed: index-2 DAE handled on ker(Cg) via polar_spectrum projections
    - sign matches S4's Lyapunov convention (not eigenvalues -- S9 forbids)
    TEST: perturbed fixed point contracts monotonically IN THE P-NORM
      (V = v^T P v, S4 Lyapunov certificate) at linear amplitude over 20
      units of s. [Bar amended 2026-08-02, EJA #72: raw-norm monotonicity
      at amp 1e-3 is unsatisfiable per the certified S6 transient.]
    FILE: M1_GATE_LADDER.out prints CONTRACTS with the S4-convention sign

M2  past the transient
    - S6 records transient growth up to 5807.6x
    - a readout is admissible only once ||dz|| has fallen BELOW its initial
      value after peaking
    TEST: the orbit's max excursion is passed and ||dz(s)|| < ||dz(0)||
    FILE: VERDICT_ORBIT.txt records s_transient

M3  the verdict
    - settling / wandering / cycle, from cos_step + V, with bootstrap CI
    - at least 3 independent seeds agreeing
    TEST: all CIs inside one regime
    FILE: VERDICT_ORBIT.txt

M4  sigma_Lambda(nu) magnitude
    - SIGN already ESTABLISHED (two grids). MAGNITUDE is not.
    TEST: cross-grid spread < 0.15 at two viscosities
    FILE: sigma_peak.py output

M5  novelty settled
    - literature kill-search on sigma_Lambda / vorticity-direction criticality
    - every novelty claim in README moves to CONFIRMED or RETRACTED
    FILE: NOVELTY.md

## Done means

ALL of M1-M5 PASS  ->  the project is complete; write it up.

OR any milestone returns a PROVEN BLOCKER -- a recorded reason it cannot be
reached with available means (like S6 was for the physical-frame march). A
blocker is a result and closes the milestone. "We ran out of resolution" is a
blocker only when the requirement is QUANTIFIED (e.g. "needs 20 e-folds, have
1.3").

## Not done, and never was

- Navier-Stokes regularity. Out of scope, stated in README.
- The strip -1/2 <= sigma <= 0. That is a theorem, not a measurement, and
  belongs to a different project.
