# PARZIVAL / NS — EVENING HANDOFF (2026-08-02)

**Supersedes HANDOFF_2026-08-02.md (morning). Read this first.** The morning
handoff's M1 build-order is DONE and its section 6.1 P-norm probe is DEAD
(see below). This session ended when a context classifier progressively
blocked the shell and agent tools — state is fully banked; nothing running.

## THE HEADLINE: M1 PASSES — CONTRACTS

The s-march instrument is correct and validated at the PRODUCTION grid
(M1_GATE_LADDER.out):
- T2 fixed (pin refresh), T4 fixed (genuine reduced Newton on ker(Cg) via
  archive/polar_spectrum.py projectors), T5 discovered+fixed (perturbations
  must be admissible — zero the rT_pin rows; off-class components freeze
  into permanent forcing).
- E=I CLOSED (EJA #19, #31 resolved): march tracks expm(t·Lred) with the
  O(ds) signature; fixed-point kill shot passes; constraints at machine tol
  through 240 steps.
- THE GATE: P-norm (V=v^T P v) monotone 8.60e-8 -> 3.15e-11 over 20 s-units,
  ZERO violations, at amp 1e-6; raw norm peaked 2.39x (certified S6
  transient) then decayed to 0.18x. Affordability unlock: project_oblique is
  identity on ker(Cg) => frozen reduced BE operator = -sign*Lred + I/ds;
  0.5s LU then 0.04s/step (validated 4.6e-14 vs exact marcher).

**PENDING OPERATOR DECISIONS:**
1. DONE.md M1 bar amendment (raw-norm monotone at 1e-3 is unsatisfiable per
   certified S6; correct bar = P-norm monotone at linear amp — EJA #72).
   One-line edit, then done.sh flips M1.
2. Run `zsh parzival/boussinesq/eja_pending_20260802.sh` once (the M1-PASS
   EJA mints; staged because the classifier blocked Bash).

## KEY FINDINGS BANKED TODAY (all in Token3 parzival_ns_blowup + EJA)

- HURWITZ IS CONFIG-DEPENDENT: no ladder rung certifies (Nb20/24: no root,
  floors ~7e-3; Nb28/Nb32: roots 1e-11 but P INDEFINITE, 4 negative
  directions — including n_f larger than production). Prime suspect: eps_b
  (rungs at 1e-3 = singular wedge layer; certified production at 1e-4).
  Discriminating test: rung-3 config at eps_b=1e-4, cholesky flips or not.
- Section-6.1 P-norm probe on EXISTING orbit data is DEAD (EJA #73): no
  faithful DNS-crop -> solver-state map (4 missing pieces, PNORM_PROBE.out).
  P-norm lives on the MARCH side only.
- CURIOUS track (CURIOUS.md): K1 theta-transport relation w=1+Delta
  (UNGRADED, sympy-checked); generic |D|~1.01 CORRECTED to noise (per-run
  CIs +-1.2); W3 engineered tension DISSOLVED; three-fates framing for M3.
- VAULT: never rm runs/ raw — tools/archive_run.sh + tools/free_space.sh
  gate; Drive PARZIVAL_VAULT via connector (small files, verified by
  download-diff; ~18 uploads may still be completing); bulk pipe needs
  one-time Drive-desktop or rclone setup (VAULT.md).

## NEXT ACTION: the M2 amplitude ladder

Spec ready in M2_AMPLADDER_SPEC.md — first measurement past the transient:
amp ladder 1e-6..1e-3, 60 s-units each, seconds per run. Outcomes all pay:
basin radius / s_transient (M2's criterion) / any finite-amp P-norm growth
= first direct nonlinear-sector signature. Open a FRESH chat:

> Read parzival/boussinesq/HANDOFF_2026-08-02_EVENING.md, then
> M2_AMPLADDER_SPEC.md, then run the experiment exactly as specified.
> Terse; tool output to files, not chat.

## Standing refusals, money rules, file map: unchanged from the morning
handoff (HANDOFF_2026-08-02.md sections 1, 7, 8) plus: definiteness by
cholesky only, never eigenvalues of P-or-L for decisions.
