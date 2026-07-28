# PARZIVAL — CONTEXT HANDOFF (updated 2026-07-25; base 2026-07-24)

> **2026-07-27 — THE ALPHA CAMPAIGN IS CLOSED: read
> `boussinesq/ALPHA_RESULT.md` FIRST.**  The corner-regularized panel solver
> (`polar_cornerreg.py`) independently reproduces alpha = −0.34240 ± 3e-5
> (quadratic eps_b→0 extrapolation −0.34240089 vs reference −0.34240009) with
> the free gauge residual at 1.8e-6.  That document carries the full mechanism
> stack, the retraction ledger, and the ranked next steps (numerics note;
> unstable branches; spectrum via resolvent).  Everything below on the profile
> solve is historical context; Token3 `parzival_ns_blowup` has the full capture.

> **2026-07-25 addendum at the bottom (section 8): the axisym degeneracy
> lattice.** It contains a RETRACTION of a caveat in the s=6 write-up and
> closes the IC search. Read it before running any more initial-condition
> scans.
>
> **ALSO 2026-07-25 — `IC_POWER` default was WRONG (1.0; PNAS eq 3a is 4).** Now
> 4.0, and every run records its IC under `res['ic']`. Runs `ax_Z3W1_*` and
> `disc_D*` predate the fix and are VOID; `ax_Z1_*`/`ax_Z3_*` passed
> `--ic-power 4` explicitly and are fine. **Always pass `--ic-power 4`.**

Supersedes the 2026-07-23 handoff, which was stale in every operational detail (it
described an AWS ladder as running and a conda install as in-flight; both long finished,
boxes gone).

Detail lives in memory: `project_parzival_blowup_result.md` (the science + every
retraction), `project_parzival_m1.md`, `project_aws_idle_burn_incident.md`.

---

## 1. MONEY / OPS — nothing is running

- **AWS is EMPTY.** Zero instances across all 17 enabled regions; zero snapshots. Swept.
- **Budget alarm LIVE:** `parzival-spend-guard`, $50/month, emails hill.jt@icloud.com at
  $25 / $50 / and on FORECAST breach (the forecast trigger is what would have caught the
  incident below in a day or two instead of 17).
- **The incident:** a g6e.xlarge named `train-A-524m` ran EMPTY for 17 days = **~$658**.
  Support case filed same day for a one-time adjustment; nothing had hit the card (AWS
  bills in arrears). **Cost Explorer is DISABLED** on the account, which is why every cost
  API returns DataUnavailable — check spend in the Billing console.
- **The guard that works:** launch with the DEFAULT shutdown behaviour and have the run
  script call `sudo shutdown -h now` on solver exit. The box STOPS itself (compute billing
  ends; EBS + results survive for pulling; terminate by hand after). Validated live — the
  N=2048 box stopped itself unattended. Do NOT self-TERMINATE: that destroys the results.
- Perspective: today's science cost **~$6**. The $658 was the empty box. Everything that
  actually decided anything ran on the laptop or on checkpoints already on disk.

---

## 2. RESULTS THAT STAND

### 2D Boussinesq (`boussinesq/dedalus_bsq.py`) — T* = 1.70 +/- 0.01
Use **theory-forced exponents, never a fitted one.** The scaling group forces
`||grad b|| ~ (T*-t)^-2`, so **`(1/sup|grad b|)^(1/2)` must be LINEAR in t**, hitting zero
at T*. Two fitted params, exponent from theory.

| run | window | T* | R^2 |
|---|---|---|---|
| N=1024 | [1.44,1.529] | 1.7076 | 0.99964 |
| N=2048 | [1.44,1.545] | 1.7033 | 0.99956 |

Resolution-independent (dT* = 0.004 across a 2x refinement); `sup|grad b|` agrees 0.0-0.2%
to t=1.505. Two self-consistency checks land inside: forcing the exponent to exactly -2
gives T*=1.696; forcing gamma to the literature 2.9206 gives 1.7014. Independently
`gamma = 2.94-3.05` from `L = ||omega||/||grad omega|| ~ (T*-t)^gamma`. BKM criterion MET
(`int ||omega|| dt` linear in `-ln(T*-t)`, slope 1.165, R^2 0.994).
**Caveat: the forced exponents are APPROACHED, not attained (-1.90 of -2.0 by t=1.53).**

### 3D axisymmetric Euler (`boussinesq/dedalus_axisym.py`) — t_s to 1.7%
Same method, different system, **pre-registered** target `TS_REF = 0.0035056`:
converged-window fit gives **0.003566 (+1.7%)**. Method now validated on two systems.
- **USE `--ic-power 4`.** PNAS eq 3a is `exp(-30(1-r^2)^4)`; the file hard-coded the FIRST
  power and an old note wrongly recorded that as a correction. Power 1 confines the swirl
  to `r in [0.983,1]`, power 4 to `[0.76,1]` — a different problem. Power 1 gave +50% error.
- Fit ONLY where resolutions agree: 256 vs 512 agree <0.1% to **t=0.0028**, then 4.9% at
  0.0030 and 16.6% at 0.0034. That restriction turned +50% into +1.7%.
- `sup|w1| ~ N^0.29` late = still unresolved past t=0.0028. Deeper is cheap
  (512x1536 = 362s on 4 ranks; 128x384 reaches 98.5% of t_s in 17s single-core).
- `r0` CLEARED as a suspect (r0=0.4 vs 0.2 -> 0.26%).
- All 5 gates pass (G3 `D(r^2 u1)/Dt = 0` pointwise **2.0e-14**). Equations verified by
  hand against axisymmetric Euler, not only by the gates.

---

## 3. RETRACTED — do not reuse. This section is the point of this document.

- **The free-exponent analyticity-strip fit** (`delta = C(T*-t)^rho` -> T*=1.7135,
  rho=2.60, "blowup favoured 100:1"). A stretched exponential with NO singularity fits the
  same 13 points as `T*=1.735, rho=2.735` — indistinguishable. The 100x was a straw man
  (2-param opponent); against an equally-complex non-singular model it collapses to ~5x.
  The "+/-0.10" was a sub-window half-range, not an error bar.
- **"Uniform refinement is hopeless / one decade costs 398x."** rho=2.6 was an EFFECTIVE
  exponent over 0.70 decades; rho_true = 1.0 or 1.5 reproduces the fit to <2%.
- **"Newton converged to a nonzero profile."** It converged to **Om = B = 0**.
- **"Fixing both scalars excludes the trivial solution."** Wrong: `Om=B=0` solves the
  system for any c_l, c_w and has a large basin.
- **"3D axisym does not reproduce blowup."** That was a well-converged solve of the WRONG
  initial condition (ic-power above).
- **"Axisym IC uses the FIRST power of (1-r^2)."** It is the FOURTH.
- **`||omega||_inf` as a global max in 2D.** omega is ODD about x=pi so `omega(corner)=0`;
  the global max sits at x~4.05 on an unrelated broad structure. This explains the old
  1.703-vs-1.74 T* gap. Use grad-b, or a corner-LOCAL omega measure.

---

## 4. THE OPEN BUILD: profile solve in LOG-POLAR

Six Cartesian-box configurations failed for ONE root cause: the far field is
`r^alpha g(beta)`, and **on a square box no pointwise edge condition can express it**
(four straight edges spanning varying radius AND angle simultaneously).

**Derivation + spec: `boussinesq/POLAR_SPEC.md`. START THERE.** Headline: with `s = ln r`
the rescaling term becomes **constant-coefficient**, `c_l*y.grad = c_l*d_s`, so
self-similarity is s-translation invariance; and the far field becomes three homogeneous
**Robin conditions on a STRAIGHT boundary** (`d_s Om = alpha Om`, `d_s B = (1+2alpha)B`,
`d_s Psi = (2+alpha)Psi`) with `g(beta)` carried by the beta basis needing no condition.
Wedge `beta in [0,pi/2]`, wall at beta=0, symmetry line at beta=pi/2.

Carry over (all verified — do not rediscover):
- **Velocity from the CLEAN gradient.** Tau lifts inside `U = skew(grad(Psi)+lift(tau))`
  inject boundary-residual artifact into the advecting velocity: `|Psi|=0.049` gave
  `|U|=18.0` (grid-scale) and killed the run in ~5 steps. Correct in the 1-wall physical
  solver; FATAL with two Chebyshev directions.
- **`B(0,0) = 0` is FORCED.** At the corner U vanishes and `c_l*y=0`, so the B equation
  collapses to `(c_l+2c_w) B(0,0) = 0` with `c_l+2c_w = 0.947`. So `B ~ d^2` there — which
  is why Chen-Hou freeze `B_y1y1(0)`, not `B(0)`.
- **Never pin `c_l`.** It is a CONSEQUENCE of the profile, not a normalization, so pinning
  it breaks NO symmetry and leaves a neutral direction => singular Jacobian, sign-flipping
  c_w, 1e6-1e7 step norms. Two scaling symmetries need TWO weighted integral gauges — that
  config was always the well-conditioned one (residuals fell 10x monotonically); its only
  defect was the `c_l -> 0` basin.
- Outflow BCs on first-order transport singularize the Jacobian (step norm 1e12 -> 10.5
  when removed). There is NO inflow boundary anywhere.
- The profile is a **SADDLE** (`c_l d_s` outward vs `u.grad` inward vs `c_w` damping; the
  profile is the exact cancellation). Newton is right precisely because it ignores
  stability. The Jacobian at a converged solution gives the STABILITY SPECTRUM — what
  separates "a profile exists" from "this is the generic blowup".

---

## 5. PROCESS LESSONS (cost the most, generalize furthest)

1. **Clean conservation is NOT a resolution check.** Three separate times a pristine
   Casimir (1e-8 to 1e-11) coexisted with a badly unresolved feature.
2. **Newton at a nondegenerate root converges QUADRATICALLY.** A residual falling by
   exactly the damping factor is the FIELD marching to zero. **Print `||field||` beside
   the residual, always.**
3. **Verify the residual AT INIT.** A 150%-residual seed guarantees a wild first step.
4. **Aggregate flags hide failures.** One "DIVERGED" boolean masked that the blowup was
   always at the first logged step. Log per-field, per-step, from step 1.
5. **Check WHERE the argmax is,** not just its value. (omega contamination; and in 2D
   argmax|grad b| migrates from interior z/Lz~0.40 to the wall only at t>=1.44.)
6. **Spectrum tail decides physical-vs-numerical:** monotone decay = physical
   under-resolution; a flat "bathtub" that RISES near k/kmax~0.98 = spectral blocking.
7. **zsh does NOT word-split variables.** Bit me FOUR times. Write args explicitly.
8. **Fit only where consecutive resolutions agree.** This one rule turned +50% into +1.7%.

---

## 6. THE LENS INSTRUMENT (`lenses/`)

Four cacheable profiles + one swappable `situation.md`; spins no subagents.
`karpathy.md` (meta-flow: you are fooling yourself and the aggregate metric is how),
`hou.md` (the mesh IS the algorithm), `chen.md` (fixed points, conditioning, rigor),
`elgindi.md` (mechanism, what is conserved, integral criteria). Rule: three separate
verdicts, never pre-merged — the DISAGREEMENT is the information. It earned its keep:
Chen's two predictions were both confirmed empirically, and Elgindi's window sweep caught
the omega contamination that closed a gap I had logged as open and unexplained.

---

## 7. ENGINE STATUS

| file | role | state |
|---|---|---|
| `dedalus_bsq.py` | 2D Boussinesq | VALIDATED tier-one. **MPI-correct** (Allreduce-MAX, rank-0 writes, bcast control; n=1 vs n=2 bit-identical). checkpoint/stream/control verified live. |
| `dedalus_axisym.py` | 3D axisym Euler | Gates 1e-14. **MPI-fixed** (4 reductions + the `_scalar` fix that had silently zeroed the trust referee on rank 0). Use `--ic-power 4`. |
| `rescale.py` | dynamic rescaling (time-march) | stable + well-conditioned, but cannot reach the fixed point (c_l -> 0). Superseded by the Newton route. |
| `profile_newton.py`, `profile_release.py`, `profile_weighted.py` | profile root-find | Cartesian box — ALL BLOCKED by far-field geometry. Keep for the diagnostic trail; do not extend. |
| `POLAR_SPEC.md` | the next build | derivation ready. **START HERE.** |

Always `mpirun -n <PHYSICAL cores>` (hyperthread ranks hurt spectral codes). 8 perf cores
on this laptop; axisym gets ~2.9x on 4 ranks.

---

## 8. ADDENDUM 2026-07-25 — THE AXISYM DEGENERACY LATTICE (IC SEARCH CLOSED)

Full doc: **`boussinesq/DEGENERACY_LATTICE.md`**. New tool: **`boussinesq/peak_geometry.py`**.

### The law (measured, 7 configs, one out-of-sample)

    ord_z omega1(t > 0) = min( ord_z omega1(0), 2q - 1 ),   q = ord_z u1

Parity does it: `u^r` is EVEN in z and `u^z` ODD, so both advection terms in
`d_t omega1 + u.grad omega1 = d_z(u1^2)` carry order >= p and can never lower it.
Only the source can, and only to `2q-1`. Amplitude is irrelevant. Confirmed at
q=1,3,5 (q=5 predicted BEFORE measuring: z^8.871) and on both injection directions.

### RETRACTION

The s=6 write-up said *"we only degenerated half the data — omega1 = psi1 = 0."*
**Wrong.** At q=3 the source generates omega1 at order 5 within a few steps. The s=6
ladder WAS the both-fields-degenerate run, at order 5 — more degenerate than Liu's
cubic. The s=6 null is stronger than I reported.

### s=4 has no axisym preimage

u1 odd ⟹ q odd ⟹ `s = 2q ∈ {2,6,10,...}`, i.e. `s ≡ 2 (mod 4)`. In Boussinesq theta
and omega are independent (2 free degeneracies); in axisym theta is tied to u1 and
omega1 is driven by `d_z(u1^2)` (1 free). **The Boussinesq–axisym analogue is NOT
faithful at the level of degeneracy classes**, in either direction.

### METHOD RULE — do not repeat this error

**t_s is not a class invariant and equal-t is not a fair comparison.** At t=0.00345,
(2,1) sits at `(t_s-t)/t_s = 0.11` but (6,1) at `0.32` — three times further from its
own singularity. Its cleaner fit, tinier spectral tails (1e-27) and 0.8% 128-vs-256
spread are all just what "further out" looks like. **Compare at equal
`(t_s-t)/t_s`.** The temporal exponent −1 is FORCED by the scaling group, so it
carries zero discriminating information. Real discriminators: argmax location,
spatial exponent `ell ~ (t_s-t)^p`, anisotropy, and strongest — **profile collapse**.

### New CLI + gotchas on `dedalus_axisym.py`

- `--wamp/--wpow` reach the `p < 2q-1` column ONLY. They cannot make omega1 more
  degenerate; the help text says so.
- `--ckpt-sim-dt` / `--ckpt-max-writes` give controlled-time snapshots (the old
  `wall_dt` + `max_writes=2` default gave two, not enough for any spatial fit).
- `w1_ratio` is **meaningless** when `wamp != 0`; use `sup_wphys_trust_end`.
- **Always pass `--run-id`** when the IC differs — the default tag is
  `axisym_N{Nz}x{Nr}_A{A}` and same-resolution runs silently overwrite each other's
  `stream_*.jsonl` and `ckpt_*`. The (6,5) 512 stream was lost this way.

### Bottom line

Every lattice point measured lands on the same forced law. That is a strongly
attracting self-similar fixed point (CHL Stage-1/Stage-2). **Stop scanning ICs.** The
live question is the SPECTRUM of the linearisation — Liu's route — which is
`boussinesq/POLAR_SPEC.md`. Section 7 already says START HERE; this addendum removes
the last reason to do anything else first.
