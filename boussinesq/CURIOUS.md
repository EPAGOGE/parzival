# CURIOUS — the full-equation side track (opened 2026-08-02)

**Why this exists.** M1 is blocked on T2+T4. Instead of only fixing the
instrument, this track points the existing machinery at the FULL equation —
knowing the solve won't close, betting that what falls out is worth more than
the attempt. The deliverable is the weird ledger below, not a proof.

**Discipline.** Everything in this file is **UNGRADED**. The standing refusals
bind here exactly as on the main track (no eigenvalue quoting, theory-forced
exponents only, every flattering number needs a free residual, do not quote the
absolute radial exponent). Nothing crosses into `NOTE_CLAIMS.md` without
passing `STANDARD.md`. Laptop-only; zero cloud spend; tool output to files.

---

## The attacks (K-ledger)

### K1 — strict two-scale ansatz vs the full Boussinesq system  [RUN 08-02]
`curious_twoscale.py` → `CURIOUS_TWOSCALE.out`. Order-counting on the
anisotropic similarity ansatz, symbolically checked. Results (UNGRADED):
- **R1** theta-transport alone forces `w = 1 + Delta` (Delta = c_fast − c_slow):
  anisotropy is paid for by faster vorticity growth. Delta=0 recovers w=1.
- **R2** the omega equation then balances identically — no extra condition.
- **R4** `w = 1 + Delta ≥ 1`: two-scale collapse is automatically
  BKM-supercritical.
- **R5** confrontation: `|D| = (w−1)/w < 1` strictly. Generic ICs measured
  `|D| ≈ 1.01` → at/above the admissibility boundary → strict two-scale
  power law likely IMPOSSIBLE for generic data (→ W2). Engineered ICs
  `|D| = 0.16` with banked w=1 violates R1 by 0.16 (→ W3, kill test attached).
- Assumptions A1–A4 flagged in the script header. A1 (strict power laws) is
  load-bearing; the s-march is the instrument that sees past it.
- **EJA'd 08-02:** arc `curious-fulleq` node 0; deductions minted (R1, R4,
  engineered-drift-zero NOVEL); refusal `lz_rate_is_half_cl` (negative
  control); tensions #67 (data gap, consumed by the rerun), #68 (A1), #69
  (K2 CF-marginality). Self-audit #70 RESOLVED: dbars v1 mislabeled a
  spans-both-boundaries CI as FORBIDDEN; verdict logic corrected.

### K1a — per-run |D| bars (`curious_dbars.py` → `CURIOUS_DBARS.out`)  [RUN 08-02]
First sweep, 13 surviving snapshot sets: every engineered/control run is
admissible with CI containing 0 (isotropic-consistent); the one generic
survivor (G5_11, n=16) is UNINFORMATIVE (CI spans both boundaries). The
decisive generic runs' fields were deleted — reruns W3R/W63R/W77R launched
(`rerun_wseeds.sh`, tmax 0.0013, snap cadence 2.5e-5, new run-ids so the
original streams are preserved); the sweep re-runs itself on completion and
drops `WSEED_RERUN.done`.

### K2 — the criterion race  [NEXT, cheap]
Rank the classical regularity criteria (BKM, Constantin–Fefferman vorticity
direction, Ladyzhenskaya–Prodi–Serrin norms) on the EXISTING blowup snapshots:
which binds last, which is marginal. Sharpest sub-question: sigma_Lambda = +1.00
(ESTABLISHED) says |grad xi| grows exactly like |omega|^{1/2} — is that
precisely CF-marginality? If yes, the banked criticality result and a named NS
regularity criterion are the same object. Uses `features.py` machinery +
existing snapshots only.

### K3 — how the corner root dies under viscosity  [MEDIUM]
Add a nu·Laplacian row to `polar_cornerreg.py`'s residual and continue the
banked root in nu from 0. Self-similar NS blowup of this leading-order type is
expected to be obstructed — the DATA is the failure mode: drift to parabolic
scaling, or a fold at some nu*. A fold would connect to the ESTABLISHED viscous
sign inversion of sigma_PEAK (→ W4). Free residuals (h_id, d_cl) come along
for free and gate every step.

### K4 — the regularization exponent T*(nu)  [MEDIUM]
`dedalus_bsq.py` with viscosity on: measure T*(nu) − T*(0) vs nu with
theory-forced exponents only. Feeds M4 (sigma_Lambda magnitude) as a side
effect. Stage runs with nohup; never wait on them.

---

## The weird ledger (each entry carries a kill test; UNGRADED until it doesn't)

**W1 — the 1.433 coincidence.** The universal axial rate −1.433 ± 0.083
(under w=1) numerically brackets `c_l/2 = 1.4603` (0.33 sigma), and also
sqrt(2) (0.23 sigma) and 3/2 (0.81 sigma). Bar too wide to discriminate —
numerology until shrunk ~3×. KILL TEST: re-measure ell_z with the integral
estimator on existing snapshots; post-M1, a long s-march window.

**W2 — generic two-scale is (probably) equation-forbidden.** K1-R5: strict
two-scale needs |D| < 1; generic measured |D| ≈ 1.01. If the per-run bar keeps
|D| ≥ 1, the equation itself refuses the two-exponent picture for generic data,
and M3's hypothesis space shrinks to {settles-to-isotropic-corner-class,
non-power-law wandering}. **OUTCOME 08-02 (reruns W3R/W63R/W77R done, #67
resolved): INSTRUMENT-LIMITED, not decisive — and a CORRECTION.** Trust
gating leaves 21/21/8 snapshots spanning ~1 e-fold (inside the S6 transient);
per-run CIs [−1.32, 1.42] / [−1.29, 0.99] / uninformative. So the 07-30
generic |D| ≈ 1.01 was a point estimate inside ±1.2 per-run noise — the
sign-inconsistency across seeds was the tell. Generic |D| has NO per-run
significance from physical-frame runs. The K1 exclusion stands as theory;
its measured input does not. Discriminating instrument: longer trusted spans
in rescaled time — the s-march (tension #71). W2 now waits on M1.

**W3 — the engineered-IC tension. DISSOLVED 08-02 (measured).** Per-run
block-bootstrap CIs on every engineered/control run contain 0 (e.g.
OR_z256r768: D = −0.174, CI [−0.257, 0.187]). The 07-30 magnitude 0.16 was
inside per-run noise; no R1 violation at banked w = 1. The planned T* refit
at −1.19 is moot. (EJA deduction `engineered_drift_zero_consistent`, NOVEL.)

**W4 — is there a nu*?** The viscous sign inversion of sigma_PEAK is
ESTABLISHED but unexplained. If K3 finds a fold at nu*, the two may be the same
structure. KILL TEST: K3 continuation; compare fold location (if any) to the
sign-inversion viscosity.

**W5 — BKM is free here.** Any strict two-scale collapse is automatically
BKM-supercritical (K1-R4). Closes a checking chore forever; nothing to kill.

---

## Relation to the graded track

M1 (T2+T4) proceeds unchanged — this file never touches `march_s.py`. The
K-track feeds M3/M5 only through grading. Note the convergence: K1's
load-bearing assumption (strict power laws) is exactly what the s-march can
test — if the march shows log-drift, A1 falls and W2's conclusion flips from
"forbidden" to "evaded." The curious track and the blocker fix want each other.
