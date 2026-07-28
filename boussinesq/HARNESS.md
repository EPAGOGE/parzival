# Boussinesq 2D harness — how the parzival discipline stack maps up a dimension

Era **B0** (bring-up). Written 2026-07-22 against measured numbers from
`bench_fft.py` on this machine (Apple M1 Pro, 16 GB, scipy 1.18.0 fp64 pocketfft,
torch 2.13.0 MPS). Re-run the bench and re-date this doc after any
scipy/torch/OS bump — the meter is part of the measurement.

**Standing constraints (inherited + hardened):**

1. fp64 numpy/scipy is the PRIMARY engine, not a reference sidecar. At 2D the
   fp64/fp32 roles from `swarm_m1.py` invert: the truth engine is the workhorse
   and any fp32 fast tier is a later, gated luxury (see §7).
2. **No dense N²xN² operator matrices.** The 1D engine's `build_mats` trick
   (dense H, L, D, G + matmul) is a 1D luxury: at N=128² the operator would be
   16384² fp64 = 2 GB. Everything is spectral-diagonal via `scipy.fft`
   dstn/dctn: derivatives, Λ-type dissipation, and the Poisson solve
   ∇²ψ = −ω are all pointwise multiplies in the sine/cosine basis.
   Corollary: `audit.py`'s dense-Jacobian BDF auditor does NOT port; see §4.
3. **No wall-clocks or randomness inside physics code paths.** rng lives in the
   driver only (seeded, seed recorded in the run note); physics functions are
   pure array-in/array-out; timing exists only in `bench_fft.py` and the
   driver's steps/s meter (same placement as `swarm_m1.py`'s `sps`).
4. **Vault freeze.** A campaign is running. This engine emits NOTHING into
   `vault/` or `runs/` until the freeze lifts. Completed-run payloads accumulate
   as JSON under `boussinesq/pending_vault/` (unhashed — the hash covers `prev`,
   so chaining happens at append time, after the freeze, against the then-head).

---

## 1. The equations and the spectral discipline

2D Boussinesq, vorticity–streamfunction form on the box [0, π]² with
no-penetration walls (ψ = 0 on ∂Ω):

    ω_t + u·∇ω = θ_x + ν ∇²ω
    θ_t + u·∇θ =        κ ∇²θ
    ∇²ψ = −ω,  u = ψ_y,  v = −ψ_x

Basis: **DST-II / DCT-II on the staggered (half-integer) grid.** Measured
reason, not taste: DST-I at power-of-two N runs on length N−1 FFTs and costs
4x (128), 12x (256), 3x (512) more than DST-II (bench, 2026-07-22). Sine
differentiates to cosine and back, so ∂x maps DST-II ↔ DCT-II with a diagonal
factor; ∇⁻² is diagonal in the double-sine basis (divide by j² + k²). The
walls this basis imposes are free-slip / stress-free (no-penetration only) —
which is what the inviscid target regime wants; an honest limitation for
viscous no-slip questions, stated here once so no note has to rediscover it.

Nonlinear terms are pseudo-spectral (products in physical space) with the
2/3-rule dealiasing mask applied in transform space.

Stiffness: explicit RK4 on ν∇² is unstable at any useful dt for ν ≳ 1e-3
(ν·dt·k²_max ≈ 33 at N=256, dt=5e-4, ν=1). The engine uses an
**integrating-factor RK4**: the diffusion factor e^{−ν(j²+k²)dt} is exact,
diagonal, costs zero transforms, and keeps rule 2 intact. In the inviscid
target regime it degenerates to plain RK4. dt is state-dependent CFL, the
`swarm_m1.py` pattern: `dt = min(DT_MAX(N), C_CFL·dx / max(1, sup|u|))` —
deterministic, no clocks, no randomness.

Regime honesty (load-bearing — this shapes fate semantics in §3):

- **ν = κ > 0 (full dissipation): global regularity is a THEOREM**
  (Cannon–DiBenedetto; Hou–Li). Any "blowup" fired there is an artifact by
  theorem. That regime is therefore the **meter-calibration regime**, never a
  discovery regime.
- Partial dissipation (ν>0, κ=0 or ν=0, κ>0): also globally regular (Chae;
  Hou–Li). Same status.
- **ν = κ = 0 with boundary: finite-time blowup from smooth data is PROVEN**
  (Chen–Hou, self-similar corner scenario). This is the ORACLE: a
  theorem-grade blowup our meters must be able to see honestly (§6).
- ν = κ = 0 without boundary (periodic/whole-space, smooth data): OPEN. This
  is where a blowup-candidate would actually be news — and where the compound
  claim in §3 carries all the weight.

## 2. Measured transform costs (bench_fft.py, 2026-07-22)

fp64 scipy.fft, warm, median of many reps. "w8" = `workers=8`.

| N    | dstn2 fwd | dstn2 fwd w8 | dstn1 fwd | batch-16 w8 (per lane) | MPS fp32 rfft2+irfft2 (per lane, batch 16) |
|------|-----------|--------------|-----------|------------------------|--------------------------------------------|
| 128² | 0.077 ms  | 0.077 ms     | 0.312 ms  | 0.016 ms               | 0.037 ms                                   |
| 256² | 0.301 ms  | 0.138 ms     | 3.699 ms  | 0.067 ms               | 0.054 ms                                   |
| 512² | 1.159 ms  | 0.288 ms     | 3.485 ms  | 0.324 ms               | 0.167 ms                                   |

Inverse DST-II costs ≈ the forward (pair-minus-forward: 0.081 / 0.317 /
1.56 ms single-threaded). `workers=8` is free real estate at 256²+ (2.2x /
4.0x) and batching 16 lanes with w8 amortizes another ~2-4x per lane.

**Derived RK4 throughput.** Two transform counts are quoted everywhere:
**T=14** (the low-count formulation this harness targets: conservative-form
nonlinearity + fused derivative transforms) and **T=40** (straight
pseudo-spectral Boussinesq, ~10 transforms/RHS x 4 stages — the conservative
planning number). Until the engine exists and its actual T is counted, budget
against T=40 and treat T=14 as upside.

Nominal dt(N) = 1e-3 · (128/N); steps to t=3 = 3000 · (N/128).

| Config                         | steps/s (T=14) | steps/s (T=40) | wall to t=3 (T=14) | wall to t=3 (T=40) |
|--------------------------------|----------------|----------------|--------------------|--------------------|
| 128² single lane, w8           | 930            | 325            | 3.2 s              | 9.2 s              |
| 256² single lane, w8           | 517            | 181            | 12 s               | 33 s               |
| 512² single lane, w8           | 248            | 87             | 48 s               | 138 s              |
| 128² batch-16, w8 (whole batch)| 280            | 98             | 11 s               | 31 s               |
| 256² batch-16, w8 (whole batch)| 66             | 23             | 90 s               | 258 s              |
| 512² batch-16, w8 (whole batch)| 12             | 4.2            | 15.5 min           | 41 min             |

(Batch rows are wall time for all 16 lanes together; per-lane amortized cost
is 16x cheaper. Add ~30% for diagnostics/meters when budgeting rungs in §5.)

Memory (fp64): one field at 512² = 2 MB; state (ω, θ) + RK4 stages + velocity
and derivative workspace ≈ 22 field-slots/lane. Batch-16 at 512² ≈ 700 MB,
batch-64 at 128² ≈ 180 MB — comfortably inside 16 GB. The binding constraint
is CPU (8 performance cores saturated by w8), not memory: bigger batches past
~16-64 add step latency, not throughput.

## 3. Lane / fate / ledger semantics at 2D

**Lane** = one (A, seed) initial condition. A is the amplitude dial scaling a
fixed IC family (e.g. θ₀ = A · buoyancy-profile); seed indexes a small,
driver-generated perturbation (rng outside physics, rule 3; both recorded in
the note). A lane owns its (ω, θ, t, dt) state and its meter channels.

**Batch sizes** (measured-realistic, not aspirational): 64 lanes at 128²,
32 at 256², 8-16 at 512². The 1D swarm's B=16384 was a matmul luxury that
does not survive the dimension jump; the ledger below is sized accordingly.

**Fates — honest for an OPEN problem.** The 1D engine could say "blowup"
almost flatly because CLM blowup is settled behavior and fp64+gates carried
the claim. At 2D inviscid the regularity question IS the science, so:

- `decay` — sup|ω| below DECAY_FRAC · (its own running peak) after t >
  DECAY_TMIN. Cheap, benign, still resolution-conditional but no one will
  fight over it.
- `fired` (**never "blowup" as a bare fact**). A fired label is a COMPOUND
  claim, all conjuncts recorded in the note:
  1. **G (growth diagnostic):** sup|ω| > M_FIRE **and** the trailing-window
     log-derivative d/dt log sup|ω| is increasing (acceleration, not
     saturation) **and** the BKM-analog integrals (∫sup|ω| dt and, per
     Chae-type criteria, ∫sup|∇θ| dt) are trending divergent.
  2. **T (trust wires green at firing time):** high-shell spectral tail
     fraction < TAIL_TRUST **and** the M2-analog budget residuals (§4) below
     the era threshold **and** dealiasing mask intact **and** dt above
     dt_min.
  The claim that gets vaulted is exactly: *"at resolution N, era Bk, with
  green meters, the growth diagnostic fired at t=T*."* Nothing more. If any
  trust wire is red at firing, the label is `escaped-meter`, which is a
  RESOLUTION event, not a physics event, and goes to the lowtrust channel.
  Promotion path: fired at N **and** at 2N with consistent T* trend and
  self-similar diagnostics → "blowup-candidate (ladder-consistent)", trust
  `quasi`. Only theorem-grade work (a rung-3-style certificate, or landing on
  the Chen–Hou proven scenario) ever makes it more.
  And by §1: any `fired` in a fully-dissipative run is auto-flagged
  ARTIFACT — the theorem outranks the meter.
- `hover` — t > HOVER_T, sup|ω| in a middle band, neither fired nor decayed.
  Same meaning as 1D: candidate shadowing of an unstable object on the fate
  boundary (the 2D analog of the rung1 edge state). Recorded via
  `record_hover`, NEVER in the fate denominator — the deflation bias caught
  in `swarm_m1.Ledger.record` (2026-07-22) applies verbatim.
- `budget-exhausted` — step budget hit with no fate. Distinct from hover
  (no band condition). Not a fate; recycled, counted.
- `stalled` — CFL drove dt below dt_min: the lane outran the meter's ability
  to integrate it within budget. Goes to lowtrust; at 2D this is a real
  channel, not a corner case, because near-fired lanes stiffen.
- `dead` — NaN/Inf poisoned. Recycle, never ledger (verbatim 1D guard:
  `is_dead` first, `inf` still counts as fired-side for the guard ordering,
  NaN comparisons are False).

**Ledger.** Same bandit skeleton as `swarm_m1.Ledger`: cells over
[A_LO, A_HI], sample → resolve → record → reweight toward the p∈(0.02, 0.98)
fate-boundary band, 0.5-crossing A* estimate with crossing-count ambiguity
reporting. Two 2D adaptations: **NC = 40** (half the 1D cell count — fate
throughput is ~100x lower, and 80 cells would starve), and the fp64-anchor
block becomes a **resolution anchor**: the primary engine already IS fp64, so
the end-of-run re-resolve of the boundary band (±1 cell) happens at **2N**
instead of in higher precision. The vaulted A* carries `[N, 2N]` bracket
semantics exactly where the 1D note carried `[fp32, fp64]`.

## 4. The audit stack (tier-Q machinery) at 2D

Gates run before science, every engine start, skippable only by explicit flag
— unchanged law.

- **Gate 1 (exact solutions).** No CLM-style closed form exists, so three
  cheaper exactness anchors, all fp64 machine-precision assertions:
  - 1a: θ ≡ 0, single Stokes eigenmode ω = sin x sin y decays exactly as
    e^{−2νt} (Euler steady state + diagonal viscosity). Tests advection +
    Poisson + IF-viscosity wiring in one shot.
  - 1b: stratified rest state θ = θ(y), u = 0: must stay exactly at rest
    while θ obeys the exact 1D heat kernel in the cosine basis. Tests that
    buoyancy torque θ_x is wired with the right sign and zero-mode handling.
  - 1c: linearized internal-wave/RT growth: small-amplitude single-mode
    perturbation of stable/unstable stratification must reproduce the
    analytic dispersion-relation growth/oscillation rate to O(amplitude²).
- **Gate 2a (implementation equivalence).** The 1D pattern (torch-fp64 vs
  numpy-fp64) becomes: the DST-II engine vs an **independent odd/even-extension
  periodic engine** (numpy.fft.rfft2 on the 2N extension — same math, disjoint
  code path). Fates identical, fields matching to ~1e-12 over a validation
  window at N=128.
- **Gate 2b (precision).** Dormant until an fp32 tier exists (§7); then:
  fates must match fp64, growth-diagnostic drift tripwired loose (the 1D
  fp32 T*-drift lesson: characterize first, tripwire at ~2x the measured
  characteristic).
- **M2 analog (live resolution meter).** Discrete residuals of the continuum
  budgets: energy d/dt ½∫|u|² = ∫θv − ν∫|∇u|², enstrophy supply vs
  dissipation, and d/dt ∫θ² = −2κ∫|∇θ|² (θ² is inviscidly conserved — the
  cleanest wire of the three). Same law as 1D: the discrete violation IS the
  aliasing error of the nonlinear term; it meters under-resolution
  mid-flight, per lane, for a few extra transforms every K steps. Thresholds
  are calibrated per era/dt/N during B0→B1 and recorded, exactly like the
  1D meter's smooth/edge/transient calibration triplet.
- **M1 analog (step tripwire).** Energy change across one IF-RK4 macro step
  vs Simpson quadrature of ⟨w, rhs⟩ over stage states, reusing stages.
  Ports with no conceptual change; recalibrate the dt² scaling per era.
- **Tier-Q cross-method auditor.** The 1D auditor's dense analytic Jacobian
  + BDF is prohibited by rule 2 (the Jacobian would be 16384² at N=128).
  Honest replacement: **scipy DOP853** (adaptive high-order explicit,
  independent error control and step machinery, Jacobian-free) re-resolves
  ~1% of boundary-cell fates on CPU. This is an integrator-family downgrade
  from implicit-multistep and is recorded as such; if a stiff viscous
  question ever matters, a matrix-free Krylov-implicit auditor is the
  upgrade path.
- **Law unchanged:** monitors are never gradients, never steering targets,
  never inputs to dt or fate logic beyond the trust wires
  (probe-is-not-the-loss).

## 5. Campaign ladder, eras, vault families, budgets

**Era tags.** This engine starts **era B0**. B-rev increments on ANY change
to meters, thresholds, dt policy, dealiasing rule, or fate constants —
results are comparable only within a B-rev. Resolution is orthogonal and
composes into the tag: `B1-N256`. Planned:

- **B0** — bring-up: gates 1a/1b/1c + 2a passing, meters wired but
  thresholds uncalibrated. Calibration runs (including deliberate
  fully-dissipative artifact-fires to exercise the ARTIFACT auto-flag) close
  the era.
- **B1** — meters calibrated, thresholds recorded; scan-eligible.
- **B2+** — earned, not scheduled.

**Vault families** (deferred to `pending_vault/` until the freeze lifts;
concept notes `boussinesq-engine` and `chenhou-scenario` created at first
emit; links into the existing graph: `[[critical-threshold]]`,
`[[hover-requires-depletion]]`, `[[swarm-engine]]`):

| Family                        | Rung | Contents |
|-------------------------------|------|----------|
| `bsq-gates-<stamp>`           | 0    | gate results + meter calibration triplets (closes B0) |
| `bsq-scenario-<name>-<stamp>` | i    | single-lane full-diagnostic runs |
| `bsq-fatescan-<stamp>`        | ii   | ledger summary, A* estimate + crossing count |
| `bsq-ladder-<stamp>`          | iii  | boundary band across N=128/256/512, era-tagged |
| `bsq-chenhou-<stamp>`         | iv   | oracle-contact runs |

All start trust `quasi`. Trust flips break the chain by construction
(inherited emit rule: hash covers trust).

**Rungs and budgets** (T=40 conservative / T=14 upside, w8, +30% diagnostics
overhead folded in):

- **(i) Single scenarios, full diagnostics.** One lane, every meter on,
  fields snapshotted on a fixed schedule. Per t=3 run: **15-45 s at 256²,
  1-3 min at 512²** (12-33 s and 48-138 s integration plus diagnostics). A
  dozen scenario runs per rev is an hour, not a day. This rung is where the
  IC families get chosen.
- **(ii) Amplitude-dial fate scan.** N=128², batch-64, NC=40 ledger.
  ~1.9 s/lane (T=40 amortized) → **400 resolved fates ≈ 13 min; 1000 ≈ 32
  min**. One evening buys a clean p(A) curve with the boundary band located.
- **(iii) Resolution ladder.** Boundary band only (±1 cell): 64 lanes at
  256² ≈ **17 min**; 16 lanes at 512² ≈ **41 min** (whole-batch walls,
  T=40). A full ladder pass — scan at 128, confirm band at 256, anchor at
  512 — is **~1.5-2 h** end to end. Era tag per rung row; the ladder note is
  the first place a `fired` label can be promoted to
  "blowup-candidate (ladder-consistent)".
- **(iv) Chen–Hou oracle contact.** 2-3 targeted runs at 512² driven toward
  the corner scenario (below), each run pushed past t=3 toward close
  approach with CFL-shrinking dt: budget **1.5-3 h** including a 256²
  ladder check. Full campaign through rung iv: **an evening to a day of
  laptop time; nothing here needs an overnight below 1024².**

## 6. What would constitute contact with the Chen–Hou oracle

The oracle's role is inverted from discovery: Chen–Hou PROVED finite-time
blowup for inviscid 2D Boussinesq with boundary (smooth data, self-similar
corner scenario descending from Luo–Hou). It is ground truth the way
`A_STAR_CPU = 5.5348` was ground truth for the 1D swarm: **if this engine
cannot see a proven blowup with green meters, its fired labels elsewhere
mean nothing.** Contact is a validation gate for the whole harness, run as
rung iv.

Pre-registration discipline (the `hover-requires-depletion` pattern): before
any contact run, transcribe the paper's certified quantities — the
self-similar scaling exponents and profile normalization — into a
`chenhou-scenario` concept note as oracle values. Do not quote them from
memory into code, and do not let a contact run be the thing that first
writes them down.

Contact = ALL of, each a recorded diagnostic, never a vibe:

1. **Geometry.** ν = κ = 0, symmetry-restricted IC (θ odd in x about the
   wall-normal axis) producing a hyperbolic stagnation point ON the
   boundary, with max|ω| attained at the boundary and the θ-front steepening
   into the corner — not an interior event.
2. **Growth.** `fired` under the full §3 compound (green meters), with the
   BKM-analog integrals trending divergent and T* stable under dt refinement.
3. **Self-similar collapse.** Rescaled profiles ω(x*/L(t))/Ω(t) collapsing
   across at least a decade of Ω growth, with fitted exponents consistent
   with the transcribed oracle values within stated ladder error.
4. **Ladder consistency.** Items 1-3 reproduced at two resolutions
   (256²/512²) with the drift between them quoted in the note.

Anything less is "approach", vaulted honestly as such. Contact earns trust
`quasi` on the note and — more importantly — retroactively licenses the §3
fired semantics for the no-boundary OPEN-regime scans, where a
ladder-consistent fired would actually matter.

## 7. MPS fast tier — measured verdict, not opinion

Measured (bench, 2026-07-22): torch 2.13 MPS fp32 `rfft2+irfft2` carries a
~0.3-0.5 ms latency floor that barely moves from 128² to 512² single-shot.
Batched x16 it reaches 0.037 / 0.054 / 0.167 ms per lane per pair vs scipy
fp64 w8 batched ≈ 0.033 / 0.14 / 0.68 ms — i.e. **~1x at 128², ~2.6x at
256², ~4x at 512²**. But the engine needs DST/DCT, which MPS lacks natively;
building them from rfft2 via odd/even extension doubles the transform length
per axis and eats most of that margin — realistic net win **~1-2x below
512², maybe ~2-3x at 512²**, purchased at the price of fp32 semantics, a
gate-2b campaign, and a second code path to keep honest.

Verdict: **do not build it for rungs i-iii** — the fp64 ladder fits in an
evening (§5) and the fp32 tier would cost more wall time to validate than it
saves. Re-open only if a post-contact campaign needs sustained 512²+ scans
(>2-3 h of batch time per rev), and then only after re-benching the actual
odd-extension composite, not the raw rfft2 proxy.

---

*Files: engine lands as `boussinesq/bsq_m1.py` (era B0), bench is
`boussinesq/bench_fft.py`, pending notes in `boussinesq/pending_vault/`.
Nothing outside `boussinesq/` is touched until the running campaign's freeze
lifts.*
