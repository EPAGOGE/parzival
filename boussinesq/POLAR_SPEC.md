# Profile problem in LOG-POLAR coordinates — derivation and spec

Why: six Cartesian-box configurations failed for one reason. The far field is
`r^alpha * g(beta)` — an algebraic radial power times an angular profile. On a square box
the "far field" is four straight edges spanning varying radius AND angle at once, so
**no pointwise condition on those edges can express it.** In polar it is ONE radial
boundary with the angular structure carried by the basis.

## Coordinates

Corner of the wedge at the singularity. Wall at `beta = 0`, symmetry line at
`beta = pi/2`:

    y1 = r cos(beta)      (along the wall, from the symmetry line)
    y2 = r sin(beta)      (away from the wall)
    domain: beta in [0, pi/2],  r in (0, R]

Use the **log-radial** variable `s = ln r` (this is the key move):

    r = e^s,      d_r = e^{-s} d_s,      s in [-S, +S]

## Operators

    Laplacian:  Lap = d_rr + (1/r) d_r + (1/r^2) d_bb  =  e^{-2s} ( d_ss + d_bb )
    advection:  u.grad = u_r d_r + (u_b/r) d_b         =  e^{-s} ( u_r d_s + u_b d_b )
    cartesian d_1:   d_1 = cos(b) d_r - (sin(b)/r) d_b =  e^{-s} ( cos b d_s - sin b d_b )

## THE PAYOFF — the rescaling term becomes CONSTANT-COEFFICIENT

    c_l * y.grad  =  c_l * r d_r  =  **c_l * d_s**

In the box this was the variable-coefficient term `c_l*y1*d1 + c_l*y2*d2` that fought
every boundary condition. In log-polar it is a plain constant-coefficient TRANSLATION in
`s`. Self-similarity becomes invariance under s-translation, which is exactly why this is
the natural coordinate system for the problem.

## The far field becomes SAYABLE

`Om ~ r^alpha g(beta)`  =>  `Om ~ e^{alpha s} g(beta)`, so at the outer radial boundary
`s = +S` the condition is a clean homogeneous Robin condition on a STRAIGHT boundary:

    d_s Om = alpha * Om          alpha = c_w/c_l
    d_s B  = (1 + 2 alpha) * B
    d_s Psi = (2 + alpha) * Psi

The angular structure `g(beta)` is carried entirely by the beta basis and needs no
condition at all. This is the thing that could not be written on the box.

## Rescaled steady system (what Newton solves)

    c_l Om_s + e^{-2s}(Psi_s Om_b - Psi_b Om_s) = c_w Om + e^{-s}(cos b B_s - sin b B_b)
    c_l B_s  + e^{-2s}(Psi_s B_b  - Psi_b B_s ) = (c_l + 2 c_w) B
    e^{-2s}(Psi_ss + Psi_bb) = -Om
    u_r = -(1/r) d_b Psi = -e^{-s} Psi_b ,  u_b = +d_r Psi = +e^{-s} Psi_s
    ==> u.grad f = e^{-2s} ( Psi_s f_b - Psi_b f_s )        [a POISSON BRACKET]

**SIGNS CORRECTED 2026-07-25 and gated: `polar_ops_gate.py` (run with
`~/parzival/.venv/bin/python3` -- needs sympy, which the dedalus venv lacks).**
The signs originally written here were BOTH WRONG for the engines' convention
`u = skew(grad psi) = (-psi_2, psi_1)` (`dedalus_bsq.py:62`), and gave exactly
`-1x` the correct advection -- which would have reversed inflow and outflow
everywhere, fatal for a saddle-type profile problem. Derivation, with
`e_r = (cos b, sin b)`, `e_b = (-sin b, cos b)`:

    u_r = -Psi_2 cos b + Psi_1 sin b = -(1/r) Psi_b
    u_b = +Psi_2 sin b + Psi_1 cos b = +Psi_r

Three identities are now verified SYMBOLICALLY (exact, identically zero) on a
manufactured non-separable field, with a control showing the old signs FAIL:

    A  u.grad f = e^{-2s} (Psi_s f_b - Psi_b f_s)     PASS
    B  Lap Psi  = e^{-2s} (Psi_ss + Psi_bb)           PASS
    C  d_1 f    = e^{-s} (cos b f_s - sin b f_b)      PASS   <- the B source in the
                                                              Om equation; a sign
                                                              error here flips the
                                                              forcing entirely

**[SUPERSEDED -- the substitution is MANDATORY, not optional. See section 3's
correction: a per-point metric gives RAW 5.1e-05 vs SUBST 1.8e-15, a factor 2.8e10.
The earlier 'RAW is not disqualified' verdict came from a GLOBAL L2 norm that over
10.8 decades cannot see the small-s end where all of RAW's error lives.]**
Substitute the known growth analytically, `Om = e^{alpha s} Ot` etc., to make
the unknowns O(1); with the Robin conditions above this may be unnecessary. Note the
Cartesian attempt at this FAILED because I imposed `dOt = 0` ("flat"), but the tilde
tends to `g(beta)`, an ANGULAR function — in log-polar that is automatic, since `d_s` and
the angular structure are cleanly separated.

## Symmetry / boundary conditions

    beta = 0      (WALL):           Psi = 0
    beta = pi/2   (SYMMETRY LINE):  Psi = 0, Om = 0 (odd LINEAR zero),
                                    B = 0 AND d_b B = 0  [DOUBLE zero, B ~ y1^2;
                                    measured Bt ~ eps^1.9992 -- see section 8.
                                    The old 'd_b B = 0 (even)' understated it.]
    s = +S        (FAR FIELD):      **ONLY Psi takes a condition here** (homogeneous
                                    Neumann d_s Pt = 0 in substituted variables).
                                    [CORRECTED: the old text said 'the three Robin
                                    conditions'. That is WRONG for Om and B. With
                                    c_l = 3.006 > 0 characteristics run OUTWARD, so
                                    s = +S is the OUTFLOW edge for the first-order
                                    transport equations, and a condition there is
                                    exactly the singular-Jacobian / 1e12-step-norm
                                    failure this document warns about below. Om and
                                    B take their conditions at s = -S. See section 8.]
    s = -S        (INNER):          characteristic/regularity. `s -> -inf` is r -> 0, the
                                    singular point itself. Truncate at a small r and
                                    impose the same Robin form (the local behaviour there
                                    is also a power law), or use the corner conditions
                                    B(0,0)=0 and Om(0,0)=0 established on the box.

## Gauge — do NOT repeat the c_l mistake

TWO scaling symmetries => TWO normalizations. **Pinning c_l breaks NEITHER** — c_l is a
consequence of the profile, not a free normalization, and pinning it leaves a neutral
direction (singular Jacobian, sign-flipping c_w, 1e6-1e7 step norms). Use the two
weighted integral gauges, which is the configuration that was always well-conditioned:

    [SUPERSEDED by the two POINT conditions -- see the CORRECTION section and
     section 6, where the point conditions are verified against Chen-Hou's own
     stored constants to 8.4e-08 and 1.2e-09. Do NOT use these integral gauges.]
    integ(wt * Om^2)         = E1      (amplitude)
    integ(wt * r^2 * Om^2)   = E2      (length scale)

with a weight that localizes on the core (unweighted integrals are dominated by the outer
boundary because `Om^2 r dr ~ r^{2alpha+1} = r^{+0.315}` GROWS).

## Carry over from the box work (all verified, do not rediscover)

- `U` must come from the CLEAN gradient. Never include tau lifts in the advecting
  velocity: `|Psi|=0.049` produced `|U|=18.0` (grid-scale) and killed the run.
- `B(0,0) = 0` is FORCED: at the corner U vanishes and `c_l*y=0`, so the B equation
  collapses to `(c_l + 2 c_w) B(0,0) = 0` with `c_l + 2 c_w = 0.947 != 0`. Hence
  `B ~ (distance)^2` at the corner — which is why Chen-Hou freeze `B_y1y1(0)`, not `B(0)`.
- Newton at a nondegenerate root converges QUADRATICALLY. A residual falling by exactly
  the damping factor each step is the FIELD marching to zero. **Always print ||field||
  next to the residual** — a small residual on a vanishing field is not a solution.
- Verify the residual AT INIT before iterating.
- The profile is a SADDLE (`c_l*d_s` pushes outward, `u.grad` pulls in, `c_w` damps, and
  the profile is the exact cancellation), so expect no basin. **[The inference that
  followed -- 'therefore Newton' -- is REFUTED; see section 10. Chen-Hou run NO
  Newton anywhere: `run_pertb.m` is an RK2 march of the rescaled equations stopping
  at 2e-10, and a recursive grep for newton|jacobian over Perturbed_eqn matches only
  a mesh-map Jacobian. Slaving c_l and c_w algebraically REMOVES the two unstable
  directions from phase space rather than stabilising them, which is the literal
  content of their title 'STABLE nearly self-similar blowup'.]** Newton was thought correct precisely
  because it does not care about stability.

---

# LITERATURE SWEEP RESULTS (2026-07-25) — read this before writing solver code

## Chen-Hou ALREADY PUBLISHED this far field, in polar, on this wedge
From the LaTeX source of arXiv:2210.07191, sec "Far-field asymptotics" (labels
`eq:ASS_asym`, `eq:ASS_decomp1`, `eq:ASS_semi1/2`, `eq:ASS_pois_1D`, `eq:normal`,
`eq:normal_pertb`). Their coordinates ARE ours: `(r,beta)` on R^2_+, `beta in [0,pi/2]`.
Their asymptotics are ours: `omega ~ g1(beta) r^alpha`, `theta ~ g2(beta) r^(1+2alpha)`,
derived by dropping l.o.t. in `c_l r d_r omega = c_omega omega + theta_x`. **In log-polar
`r d_r = d_s`, so our constant-coefficient observation IS their derivation** — the
log-polar plan is the coordinate system in which their far-field analysis is trivial.
`alpha = c_omega/c_l = -0.34240` = exactly our -0.3424.

**STEAL FIRST — the far-field angular ODE.** The outer stream function is
`r^(2+alpha) f(beta)` with f solving a LINEAR 1D Dirichlet problem on our wedge:

    (-d_beta^2 - (2+alpha)^2) f(beta) = g1(beta),     f(0) = f(pi/2) = 0

A tridiagonal solve. **Do not guess Robin coefficients — derive them from this.**
FREE NONRESONANCE CHECK: Dirichlet eigenvalues of `-d_beta^2` on [0,pi/2] are 4, 16, 36;
`(2+alpha)^2 = 2.7476`, so uniquely solvable with margin — **but it hits the first
eigenvalue exactly at alpha = 0.** If continuation drives alpha toward 0 the OUTER problem
goes singular and the solver dies for a reason unrelated to Newton. Watch alpha.

**Their far field is an ADDITIVE SPLIT, not a boundary condition:**
`omega = chi(r) r^alpha g1(beta) + omega_2` with chi a radial cutoff and omega_2 a
compactly-supported B-spline remainder. In log-polar the growing part becomes a bounded
remainder with a plain decay condition at s_max.

**THE NUMBER THAT RETRO-EXPLAINS OUR SIX FAILURES:** they compute on a box of side
**L ~ 1e13**, stream function supported to ~1e15, adaptive mesh, PLUS the semi-analytic
split — because `omega ~ r^(-1/3)` decays too slowly to truncate. **They never found a
boundary condition. They bought 13 decades of domain and factored the tail out.** On an
ordinary box no pointwise edge condition exists.

## ★★★ s>=4 REGIME: WE WERE SCOOPED THREE MONTHS AGO — and it explains our result

**arXiv:2604.01868 (Chen, Huang, Li — 2 Apr 2026).** 2D Boussinesq, half-plane with solid
boundary, odd DEGENERATE initial data (their IC is far deeper than ours:
`Omega = 10 x1^9/(x1^10 + x2^10 + 16)`, so `Theta ~ x1^10` vs our s=4). They REPRODUCE both
of Liu's qualitative findings — Omega singular at a boundary point, Theta developing a
"strict Heaviside-type jump" on x2=0 — and then **REFUTE his two-scale reading**: *"we find
no numerical evidence supporting a two-scale blowup mechanism."* Replacement is **TWO-STAGE**:
- **Stage 1**: asymptotically self-similar L-inf blowup, **REGULAR** profile, at boundary
  point **(1,0) AWAY FROM THE ORIGIN**.
- **Stage 2**: weak continuation to a local L^4 blowup **at the origin**, **SINGULAR**
  profile, `c_l/c_omega -> -2` (they report **-2.000048**).
They cite Liu's thesis 10x. **Must cite 2604.01868 or a referee will.**
Upstream theory: **Huang-Qin-Wang, arXiv:2401.14615 = SIAM J Math Anal 57(4):4068-4096
(2025)**, which states Liu's degeneracy condition verbatim and the rule "less degenerate
data -> one-scale; more degenerate -> multi-scale" (but solves the 1D CLM model).

**TERMINOLOGY TRAP:** Chen-Hou's "two-scale dynamic rescaling" (2210.07191, 2305.05660,
2405.10916) is a NUMERICAL DEVICE (separate C_l and C_omega for space and amplitude) in the
s=2 line — **NOT** a two-scale singularity. Do not read those as prior art here.

### OUR s=4 RUN, AND WHY IT "FAILED"
Ran Liu's s=4 IC ported to our engine (`dedalus_bsq.py --ic s4`), N=256 vs 512, to t=2:
| | s=2 | s=4 |
|---|---|---|
| R^2 of `(1/sup|grad b|)^(1/2)` linear fit | **0.9996** | **0.017-0.91**, erratic |
| T* across windows | 1.7076 / 1.7033 | -0.65, 2.21, 3.37, 8.72, **-164** |
| resolution agreement | 0.2% to t=1.505 | **1% only to t=0.5** (63-96% beyond) |
| max b^2 drift | 1.5e-11 | **1.2e-03** (exceeds the 1e-3 trust line) |

**THE EXPLANATION, from the checkpoints:** `argmax|grad b|` at t=1.3629 sits on the WALL at
**x/pi = 0.9297** — i.e. at distance ~0.22 from the corner, **NOT at the corner x=pi**.
That is exactly CHL's Stage-1 boundary-point blowup. **The forced -2 law is derived for
CORNER-CENTRED self-similar scaling; we applied it to a global sup tracking a structure
away from the corner.** The law was not wrong and the physics was not unresolvable — the
instrument was pointed at the wrong centre. (Caveat: only 2 checkpoints, so "consistent
with", not established.)

### WHERE A REAL GAP REMAINS (per the sweep, adversarially checked)
1. **CHL's numerics are dissipative exactly where their novel claim lives.** WENO5
   (a SHOCK-CAPTURING scheme, which will manufacture a clean Heaviside jump whether or not
   one exists) + SSP-RK3, in a dynamic-rescaling frame, and Stage 2 computed on a fixed mesh
   **with explicit numerical regularization "to suppress the growth of ||Omega||_inf and
   ||Theta_x||_inf"** — i.e. artificial dissipation applied to the weak continuation that is
   the paper's central novelty. Their resolution study is 128->1024 with L-inf
   adjacent-resolution differences of 3.9e-3 (Omega) and 6.4e-2 (V) at 512->1024.
2. **No high-order DIRECT (physical-space) simulation exists.** Liu: 1st-order upwind,
   direct. CHL: WENO5, rescaled frame. Stage 1 blowing up at X=(1,0) rather than the origin
   may be a fixture of their normalization (they pin Omega's maximizer at X=1 to concentrate
   the adaptive mesh there). A high-order direct run is the missing third leg.
3. **3D axisymmetric Euler with swirl at a boundary, s>=4: NOBODY HAS TOUCHED IT.** Liu did
   Boussinesq only; CHL did 1D HL + 2D Boussinesq only. CHL's own closing remark asks whether
   their singular profiles are "fundamentally linked" to Hou-Huang's two-scale travelling
   wave and calls it "an intriguing question for further exploration." Our axisym engine is
   gated to 1e-14 with pointwise angular-momentum conservation to 2e-14.

   **>>> ADDRESSED 2026-07-25 — see `DEGENERACY_LATTICE.md`.** This gap is now partly
   closed, and the answer is largely negative, which is itself the content:
   - `s = 4` is **FORBIDDEN** in axisym. u1 odd forces `q = ord_z u1` odd and `s = 2q`, so
     `s in {2,6,10,...}`, i.e. `s = 2 (mod 4)`. Liu's s=4 family has no axisym preimage.
   - The two degeneracy orders are not independent:
     `ord_z omega1(t>0) = min(ord_z omega1(0), 2q-1)` (measured, 7 configs, R^2>=0.9994,
     with q=5 -> z^8.871 predicted before measuring). So Boussinesq's 2 free degeneracies
     collapse to ~1 in axisym. **The analogue is NOT faithful at the level of degeneracy
     classes** — which undercuts, in both directions, the transfer of any s>=4 Boussinesq
     conclusion (Liu's or CHL's) to 3D.
   - Every lattice point run so far ((2,1), (6,5), (6,1)) lands on the same forced law.
     Degeneracy is INERT here, consistent with a strongly attracting fixed point.
   Net effect on this spec: it REMOVES the temptation to answer gap 3 with more IC scans,
   and leaves the spectrum (below) as the only live route.

### ENGINE CHANGE: log WHERE the max is, not just its value
`dedalus_bsq.py` now streams `argmax_x, argmax_z, argmax_x_over_pi, argmax_z_over_Lz`
(MPI-safe: location taken from the rank that owns the global max). Needed THREE times in one
session — (a) s=2 argmax migrates interior -> wall at t~1.44 so fits must start there;
(b) `||omega||_inf` is attained on an unrelated broad structure because omega is odd at the
corner, which explained a 1.703-vs-1.74 T* gap; (c) the s=4 off-corner blowup above.
**A max-norm diagnostic without its location is a trap.**

## ★★★ LIU THESIS Sec 3.3-3.5 — answers the exact problems we fought all session
Full PDF: `refs/Liu_thesis_FULL.pdf` (host is **thesis.caltech.edu**, NOT
thesis.library.caltech.edu — that hostname does not resolve; direct path `/9920/18/`).
A reader may report "20 pages"; that is a display limit, the full body is there.

**1. THE POISSON BC PROBLEM: DO NOT SOLVE FOR Psi.** Liu p.87, verbatim: *"To compute the
velocity field v = grad^perp (-Lap)^{-1} omega, WE CANNOT SOLVE A POISSON EQUATION to get
the stream function psi ... because ... WE DO NOT HAVE APPROPRIATE BOUNDARY CONDITIONS for
the stream function psi ... We instead employ the EXPLICIT INTEGRATION FORMULA (3.3.8c),
(3.3.8d) to compute the velocity field."* They recover omega as a piecewise bilinear
function on the nodes and integrate Biot-Savart directly. **Six of our configurations died
trying to invent a far-field condition for Psi. The correct move is to eliminate Psi.**

**2. OUR CHARACTERISTIC ANALYSIS IS CONFIRMED VERBATIM.** Liu p.86: *"Close to the steady
state of the dynamic rescaling equations, THERE IS NO INCOMING CHARACTERISTICS on the
boundary of the computational domain, so we DO NOT NEED ADDITIONAL BOUNDARY CONDITIONS for
theta~(x,t) or omega~(x,t)."* Exactly what we derived independently.

**3. FAR-FIELD TRUNCATION OPERATOR** (3.3.11), used instead of BCs:
`P_M f(x1,x2) = f - (x1^2/M^2) f(M,x2) - (x2^2/M^2) f(x1,M) + (x1^2 x2^2/M^4) f(M,M)`
applied inside the omega~ equation (3.3.12a).

**4. THE GAUGE, IN ITS SIMPLEST FORM** (3.4.3): the Jacobian is
`J = [[grad_w F_w, grad_th F_w],[grad_w F_th, grad_th F_th]]` and *"Since theta~(0,0) =
omega~(0,0) = 1 and they remain constant in the ODE, WE DO NOT VIEW THEM AS DEGREES OF
FREEDOM in computing (3.3.14) and (3.4.3)."* **Neutral directions are removed by dropping
the normalized values from the state vector** — the function-space restriction, minimal
version. Fig 3.3: *"the real parts of the first several eigenvalues ... are all negative"*.

**5. IC FOURTH POWER, CITED** (Sec 3.4): `u1(r,z) = 100 exp(-30(1-r^2)^4) sin(2 pi z/L)`.
Note the correspondence: theta in Boussinesq <-> u1^2 in Euler, so this IC gives leading
order **s = 2** for theta in x1.

**6. TWO BRANCHES, INDEXED BY THE INITIAL DATA (not by a discrete eigenvalue ladder):**
| leading order s of theta in x1 | outcome |
|---|---|
| **s = 2** | stable self-similar singularity (Luo-Hou); profiles converge; Jacobian eigenvalues negative |
| **s >= 4** | *"spatial profiles ... DO NOT CONVERGE ... do not develop self-similar singularity"*; w develops a delta-like shape, theta a jump; **MULTI-SCALE** |
CORRECTION to an earlier note here: Sec 3.5 is NOT a second self-similar profile. The
profile FAILS TO CONVERGE and a smaller scale is generated. Do not call it "a singular
profile branch". s>=4 IC used: `w = sin^3(pi x1)(1-x2)^3, theta = (1-cos(pi x1))^2 (1-x2)^2`,
DNS on `(-1,1)x(0,1)`, effective mesh `2^-18 x 2^-18`.

**7. HIS MESH** (3.4.1): half the x1 points uniform in the near field [0,10]; in the far
field `x1^i = exp(log(x1^{N1/2}) + h (i - N1/2)^{1.5})` — exponential in `i^{1.5}`, i.e.
even faster than geometric. Coarse N1=N2=40, fine N1=N2=80. Upwind in space, forward Euler
in time, CFL 0.5.

**8. HIS EXPONENTS:** `c_w = -1.4295 (N=40), -1.4281 (N=80)`; `c_l = 2.7982 / 2.8009`,
versus Luo-Hou DNS range `(2.7395, 2.9133)` and Chen-Hou's later `2.9206`.

## ★★ FORMULATION VALIDATED — both gates PASS (`boussinesq/angular_gate.py`)

Ran against Chen-Hou's own 620x620 profile. Artifact: `runs/angular_gate.npz`.

    GATE 1  angular ODE residual (relative): 1.654e-13   PASS
            f(0) = 1.4e-14,  f(pi/2) = 0                 BCs to roundoff
    GATE 2  decay exponent on FIVE independent rays (target -0.34240):
            beta = 0.20 / 0.40 / 0.80 / 1.20 / 1.40
            slope= -0.34240 / -0.34240 / -0.34240 / -0.34240 / -0.34238

**Five decimal places on five independent rays**: `r^alpha` holds UNIFORMLY IN ANGLE at
exactly the published alpha, and the angular Dirichlet problem
`(-d_bb - (2+alpha)^2) f = g1`, `f(0)=f(pi/2)=0` solves to roundoff. Nonresonance margin
`p^2 = 2.747637` vs first Dirichlet eigenvalue 4 => margin **1.2524** (singular only at
alpha=0). Extracted g1: wall = 1.125, max = 4.8911, axis = 0.0470; solved f: max = 1.4761.
**The asymptotic content of the log-polar formulation is verified against the
authoritative profile.** This is no longer a plan.

## THE ANALYTIC FAR-FIELD FORM (Sec 7.2.1) — from `FitWb.m` line 43

    g1(beta) = a1*(pi/2-beta)*(1 + a5*(pi/2-beta)^2)
               / ( ((pi/2-beta)^2 + a2)^(2/3) + a3*(pi/2-beta)^2 + a4 )

StartPoint ~ [0.7275, 0.03069, 0.1490, 0.7720, 0.2994].
- The **explicit `(pi/2-beta)` factor** hard-codes LINEAR vanishing at the axis —
  independently confirmed by our measurement (slope 1.0000 over three decades).
- The **fractional power 2/3 = -2*alpha** puts the blowup exponent into the ANGULAR shape,
  not just the radial decay. That is where an AGJ-type singularity would live. It is
  regularized twice: `a2 = 0.031` inside the fractional term and `a4 = 0.772` as an
  additive floor. With `a4 != 0` the axis behaviour stays linear regardless, so THIS
  BRANCH IS REGULAR BY CONSTRUCTION — consistent with our measurement and suggesting the
  singular profiles are a DIFFERENT member of the family.
- **Wall value enforced exactly:** `c1 = mean(w(:,1).*x1.^al)` is g1 at beta=0, and `a3` is
  then ALGEBRAICALLY RESET (lines 65-66) so the ansatz reproduces c1. Least squares in the
  interior + a hard matching condition at the wall. Copy this pattern.
- Hard quality gate: `if gof.rsquare < 0.999, error('Approximation not accurate !')`.
- **SHORTCUT:** this is a 5-parameter analytic ansatz for g1. The first outer build may not
  need to solve for the angular profile at all — fit 5 coefficients, then f follows from one
  tridiagonal solve.

## CORRECTION: `Linear_stability/` is NOT a spectrum computation

It is the **nonlinear ENERGY-ESTIMATE** machinery for the computer-assisted proof —
weighted L^inf and C^{1/2} inequality verification under interval arithmetic
(`Linear_Linf_clos_itl.m`, `Nonlin_hol_main_itl.m`, `Linear_nhol_main_itl.m`). No
eigenvalues. There are TWO distinct stability routes and they must not be conflated:
| route | method | who |
|---|---|---|
| rigorous nonlinear | weighted energy estimates + interval arithmetic | Chen-Hou (`Linear_stability/`) |
| numerical linear | eigenvalues of the discretized linearized operator | **Liu, thesis Sec 3.4** |
Our Jacobian-spectrum plan follows **LIU**, not Chen-Hou — so `refs/Liu_Pengfei_2017.pdf`
is load-bearing, not optional. Chen-Hou deliberately avoided a spectrum because energy
estimates are what interval arithmetic can certify.
Also confirmed by their Readme: *"/Lin_evo_data, /Integral_data containing large data are
not needed"*. And `Nonlin_asym.m` verifies inequalities specifically for x in the FAR FIELD.

## ★ MEASURED IN THEIR DATA — the number that explains all six Cartesian failures

Loaded `refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat`
(620x620, scipy.io.loadmat, NOT v7.3/HDF5) and measured directly:

**1. Their mesh IS log-uniform.** `Mesh.x` = two 620-pt meshes, range [0, 3.13e15];
consecutive ratios `X[i+1]/X[i] = 1.12, 1.12, 1.12, ...` exactly. Confirmed in the data,
not inferred. This mesh: `near=300, rate=[1.03,1.12], exp=[90,320]` ->
`ln(1.03)*90 + ln(1.12)*320 = 2.66 + 36.27 = 38.9` in s -> reach `e^38.9 ~ 8e16`.

**2. The far field power law is exact — and starts LATE.** |w| along the diagonal
(beta=pi/4), fitted as `log|w|` vs `log r`:

| window in r | measured slope |
|---|---|
| [1e2, 1e8] | **-0.18897**  <- PRE-asymptotic, nowhere near alpha |
| [1e5, 1e14] | -0.34275 |
| [1e8, 1e16] | **-0.34239** <- asymptotic |
| target alpha = cw/cl | **-0.34240** |

**Five-decimal agreement, measured from their raw field.** But note where the power law
BEGINS: **r ~ 1e8**. Below that the local slope is -0.189. The far field needs ~8 decades
just to ENTER the asymptotic regime.

**3. THEREFORE: `Ybox = 8` was hopeless by ~7 orders of magnitude.** In log-radius
`s_max = ln 8 = 2.08`, while the asymptotic window is **s in [18.4, 36.8]**. There was no
power law anywhere in our domain to impose a condition on. No Robin/homogeneity/matching
condition could have worked — the object it describes did not exist in the box. This is the
quantitative root cause of all six failures, and it is a DOMAIN SIZE fact, not a
formulation fact.

**4. Sizing rule for the log-polar build:** reach `s_max ~ 37` (r ~ 1e16). At
`Delta s = ln(1.12) = 0.11333` that is **326 levels** — exactly their `exp = 320`. Their mesh
is sized precisely to the asymptotic window. Put the matching/Robin boundary at
`s_match >~ 18.4` (r >~ 1e8) and NEVER below it. Angular mesh: `Mesh.xag` = **79 points on
[0, pi/2]** (their far-field angular grid).

**5. The seed carries everything.** `solu.cl = +3.00649798`, `solu.cw = -1.02942519`,
`solu.al = 0.34240009`; `rec = [3.00649798, -1.02942519, 1.19620315, 0.89909559,
-2.53267418, 0.99998786, 0.9999975]` where `rec[4] = u_x(0) = -2.53267418` (published
-2.532674) and `rec[5], rec[6] ~ 1` are the two FROZEN normalizations (their gauge, visible
as data). `solu.ag_coe` = the angular profile g1(beta) as B-spline coefficients for w and v
(bounded, peaking ~0.0554 then settling ~0.0193 — worth checking against the
Abe-Ginsberg-Jeong singular-profile claim). Also present: `u1, u2, th, v, w, w2coe, vcoe,
pcoe, xy0`. CAUTION: `pcoe` reaches ~5e27 (stream-function coefficients over 1e16 in radius)
— any port must handle that scaling explicitly.

## READ FROM CHEN-HOU SOURCE (`refs/chen_hou/Perturbed_eqn/`) — supersedes LaTeX inferences

**THEIR PRODUCTION MESH IS UNIFORM IN LOG-RADIUS.** `Para_mesh_22.m` (App C.1, Paper II):
`n1 = 355` near-field points, `rate = [1.025, 1.12]` GEOMETRIC growth ratios,
`exp = [90, 320]` levels at each rate, `pow = 2`, `NN = [58, 38]`.
A geometric mesh with ratio q means `r_{k+1}/r_k = q`, i.e. **`Delta s = ln q` = CONSTANT**:

| stage | Delta s = ln(rate) | levels | s-span | r-factor |
|---|---|---|---|---|
| slow | ln 1.025 = 0.02469 | 90 | 2.22 | x9.2 |
| fast | ln 1.12 = 0.11333 | 320 | 36.27 | x5.6e15 |
| total | — | **410** | **38.5** | **~5e16** |

So the "L ~ 1e13, stream function to 1e15" is NOT a giant uniform box — it is **410 uniform
log-radial levels**. **Their adaptive mesh IS a log-polar mesh written in physical r.** Our
log-polar plan is not a reformulation of their method; it is their method with the
coordinate made explicit. Target `Delta s ~ 0.113` and ~410 levels to match their reach.

**SIGN CONVENTION (record it, do not flip it).** In their code `al = abs(cwb/clb)` is
POSITIVE = `-alpha` = +0.3424 (`Profile_farfit.m`), and `Wangle.m` line 8 does
`wr = w .* r.^(al)` to EXTRACT the angular part. Hence their angular ODE reads
`f1 = (H1 - (2-al)^2 * L2) \ (B1*g1)` — with `(2-al)^2 = (2+alpha)^2 = 2.7476`, identical
to the form given above. Our derivation is confirmed against their source.

**g1 IS NOT SOLVED FOR — IT IS FIT FROM THE COMPUTED FIELD.** `Wangle.m` obtains g1 by
RIDGE REGRESSION (`ep = 0.05`, `g1 = (rr'rr + ep I) \ (rr'(q'Y))`) off the numerical
solution, and only THEN solves the Dirichlet problem for f1. This independently confirms the
derivation above: **g1 is undetermined at leading order in the far field and is fixed by the
INNER solution.** Any plan that tries to determine g1 from the outer equations alone is
wrong. `Profile_farfit.m` is the driver: fit the far field analytically (Sec 7.2.1 Paper I,
via `FitWb`/`FitTheta`), then re-fit with 8th-order B-splines (`LSB8_high`), then the same
angular Poisson solve for the stream-function coefficients.

**NOT the production mesh:** `Ymesh.m` self-documents as *"History mesh only used for remesh
... You DO NOT need to verify it. NOT used in verification."* Ignore it.

**Separability, useful:** `Perturbed_eqn` is the NUMERICS half and is explicitly excluded
from the rigorous verification (*"we perform a-posteriori estimates, this construction is
NOT involved in the rigorous verifications"*). So it can be read and ported without touching
the proof machinery.

**Still to fetch (TWO files only, not the 1.7 TB):** `ASS_data/` contains
`Steady_state_pertb720_Nlevcor4.mat` (the approximate steady state = **Newton's initial
guess**) and `ASS_748_Nlevcor4_Mode_e.mat`. The Readme's own init instruction is to keep
exactly those two. `Lin_evo_data` is 1.7 TB of HPC shards whose results are ALREADY combined
into `ASS_720_errfull_T0T21_Nlevcor4_itl2400.mat` + `ASS_720_bdfull_...` in `ASS_data` —
never download the shards. Also check `Perturbed_eqn/Computed profile/` first; it may already
hold a usable seed locally.

## CORRECTION: "never pin c_l" was WRONG
For the LINEARIZED/spectral problem Chen-Hou pin **`c_l == 0`** exactly (`eq:normal_pertb`:
`c_l = 0`, `c_omega = u_x(0)`, `c_theta = c_l + 2 c_omega`). The neutral direction is
removed **by RESTRICTING THE FUNCTION SPACE**, not by adding a gauge row: the perturbation
must vanish quadratically at the origin (`eq:normal_vanish`: `omega = O(|x|^2)`,
`theta_x = O(|x|^2)`, `theta_y = O(|x|^2)`). Build the Jacobian on that subspace and the
neutral mode is not in the space at all — strictly better conditioned than a bordered
system. Our diagnosis of the SYMPTOM was right; the prescription was wrong.
Their NONLINEAR gauge is two POINT conditions, not integrals (`eq:normal`):
`c_l = 2 theta_xx(0)/omega_x(0)`, `c_omega = c_l/2 + u_x(0)`, with
`c_theta = c_l + 2 c_omega` a CONSTRAINT (so exactly two scaling DOF). Equivalently it
freezes `theta_xx(0)` and `omega_x(0)` for all time.
**BONUS validation:** from `c_theta = c_l + 2 c_omega` ALONE, `grad theta` scales as
`(T-t)^-2` INDEPENDENT of the values of c_l, c_omega. Our exponent 2 is forced by the
Boussinesq structure itself — which is exactly why the parameter-free fit worked.

## DROP the BKM line
Under the forced `omega ~ (T-t)^-1`, `int ||omega|| dt` diverges LOGARITHMICALLY BY
CONSTRUCTION — the weakest possible numerical signal, and precisely what Hou-Li used to
argue AGAINST Kerr. The exponent-forced linear fit does all the work; citing BKM alongside
it weakens the case rather than strengthening it.

## REGULARITY IS THE SELECTION PRINCIPLE, NOT A BOUNDARY CONDITION
**Abe-Ginsberg-Jeong** (arXiv:2410.21765 = ARMA 250:41) treat the far-field ANGULAR profile
as an object in its own right and find that in the relevant exponent window **it is
SINGULAR — precisely on {x1=0} U {x2=0}**, i.e. the symmetry axis and the wall: **exactly
the two edges of our wedge.** The Boussinesq/axisym analogy's error terms live on those
same two rays. So we may have no right to impose `f(0)=f(pi/2)=0`.
**Where the regularity goes: into the EIGENVALUE.** Do not impose edge regularity as a
boundary condition; ADMIT a weighted class at the edges and require the solution to lie in
the right class at BOTH edges simultaneously. That over-determines the problem unless
`alpha` takes special values => **alpha is QUANTIZED**. This is the same structure as
Buckmaster-Cao-Labora-Gomez-Serrano's imploding solutions (arXiv:2208.09445 = Forum Math Pi
13:e6), where smoothness at the SONIC POINT selects discrete r; and formally identical to
quantum mechanics, where you do not impose normalizability, you REQUIRE it and the energy
quantizes. It also explains, in one stroke, Chen-Hou's nonresonance condition, the observed
NON-UNIQUENESS (Eggers-Fontelos Nonlinearity 33:325 (2020); Campolina-Simonnet-Thalabard
arXiv:2501.07377), and why DEFLATION is needed. **"Compute the profile" is the wrong
sentence; "compute the branch, ordered by instability count" is the right one.**

## Code to consult BEFORE writing more
- **Chen-Hou MATLAB code EXISTS** (their arXiv:2210.07191 comment links it; also
  jiajiechen94.github.io/codes). Authoritative, on our equations. GO HERE FIRST — it
  settles the mesh/gauge/far-field questions empirically and supplies a numerical profile
  to use as the Newton INITIAL GUESS, which is the actual blocker for a non-unique family.
- `uniFabiB/selfSimilarEulerBoussinesq` — Chebfun/MATLAB Newton for self-similar profiles
  of BOTH our systems. Steal the DAMPED NEWTON linesearch and the DEFLATION wrapper.
  (Boussinesq path unfinished, no far-field scheme — consistent with our six failures.)
- `Joel-Dahne/CGL.jl` (arXiv:2410.05480) — the architecture: NO pointwise outer BC. Inner
  solve + the KNOWN outer asymptotic evaluated analytically + C^1 MATCHING at an
  intermediate radius as extra scalar equations. In log-polar, matching at `s_match` is
  matching on a straight line. Branch continuation via BifurcationKit.jl.
- `karlesmarin/Sabra-blowup-selection` — profile Newton + Jacobian spectrum end to end,
  with explicit gauge/neutral-mode bookkeeping. Reference impl for part two.
- **Dedalus v3 already has both halves:** NLBVP (symbolic Frechet derivative — no
  hand-coded Jacobian) and EigenvalueProblem for the spectrum. Reuse the SAME
  linearization for both; that is the whole efficiency win.
- **Pengfei Liu, Caltech PhD thesis 2017** (thesis.caltech.edu/9920/) — already computed
  the linear stability spectrum for this system; Chen-Hou cite his sec 3.4 for eigenvalues
  with negative real parts bounded away from 0. Highest-value single source for part two.

## Lineage (answers "where did the pose come from")
- Method is NOT fluid-native: dynamic rescaling comes from NLS —
  McLaughlin-Papanicolaou-Sulem-Sulem, Phys Rev A 34:1200 (1986); Landman et al,
  Phys Rev A 38:3837 (1988). Chen-Hou cite these verbatim as its origin.
  **The profile-Newton-then-spectrum program was already executed for NLS in 1991-92**
  (Landman et al, Physica D 47:393). New only for this pose.
- Pose's birth certificate: **Pumir-Siggia, Phys Fluids A 4:1472 (1992)** — a NUMERICS
  paper. States the 2D-convection isomorphism away from the axis AND reports vorticity
  diverging as **inverse time squared** in 1992 (the same exponent 2 we measured).
- **E-Shu, Phys Fluids 6:49 (1994)** refuted the cap scenario and identified the SIDE.
- **Kerr, Phys Fluids A 5:1725 (1993)** vs **Hou-Li, J Nonlin Sci 16:639 (2006)** — interior
  scenarios dissolved under better resolution; **that dispute is why the field moved to a
  boundary.** Then 2014: Luo-Hou's wall (PNAS 111:12968) and Kiselev-Sverak boundary growth
  (Ann Math 180:1205), independently, same year.
- Analogy canonized in Majda-Bertozzi, *Vorticity and Incompressible Flow* (CUP 2002) —
  and stated **away from the symmetry axis**, which is where AGJ's singularities live.
- Survey to own: Drivas-Elgindi, EMS Surveys Math Sci 10:1-100 (2023), arXiv:2203.17221.
- CORRECTION: arXiv:2509.14185 does NOT include Hou as an author.

---

## Scoring targets (NOT inputs)

    c_l = 3.00649898, c_w = -1.02942516, u_x(0) = -2.532674,
    gamma = -c_l/c_w = 2.9205600           [Chen-Hou arXiv:2210.07191 eq 2.23]
    gamma = 2.91                            [Luo-Hou PNAS 111(36):12968 (2014)]
    alpha = c_w/c_l = -0.3424,  g1(beta) odd about beta = pi/2

---

# GATES CLEARED 2026-07-25 — the solver's prerequisites are now verified

Three gates, in the order the solver depends on them. Run the sympy one with
`~/parzival/.venv/bin/python3` (the dedalus venv has no sympy); the others with
`~/parzival/.venv-dedalus/bin/python3`.

## 1. `polar_ops_gate.py` — operator identities, SYMBOLIC (exact)

| identity | status |
|---|---|
| `u.grad f = e^{-2s}(Psi_s f_b - Psi_b f_s)` | PASS, identically 0 |
| `Lap Psi = e^{-2s}(Psi_ss + Psi_bb)` | PASS, identically 0 |
| `d_1 f = e^{-s}(cos b f_s - sin b f_b)` | PASS, identically 0 |

**It caught a real sign error in this very document.** The `u_r`/`u_b` lines as
originally written were both wrong for the engines' `u = skew(grad psi) = (-psi_2,
psi_1)` convention, and gave **exactly `-1x` the correct advection** — inflow and
outflow reversed everywhere, fatal for a saddle-type profile problem. The gate carries
a control that re-runs the OLD signs and confirms they FAIL, so it is not vacuous.

## 2. `polar_bc_gate.py` — far-field exponents, vs Chen-Hou's own 620x620 profile

`alpha = c_w/c_l = -0.34240009` reproduces their stored `al` exactly. Then, 4/4 to ~1e-4:

| condition | want | got |
|---|---|---|
| `d_s Om = alpha Om` | r^-0.34240 | r^-0.34239 |
| `d_s B = (1+2alpha) B` | r^+0.31520 | r^+0.31532 |
| `d_s Psi = (2+alpha) Psi` (via u ~ r^{1+alpha}) | r^+0.65760 | r^+0.65752 |
| bonus: `theta_x ~ r^{2 alpha}` | r^-0.68480 | r^-0.68468 |

**`B` IS `theta`, not `theta_x`** — read it off this spec's own B equation, whose RHS
coefficient is `(c_l + 2 c_w) = c_theta`; consistent with the Om source being `d_1 B`
and Boussinesq being `omega_t + u.grad omega = theta_x`. Their `.mat` maps as
`w`=vorticity, `th`=theta=B, `v`=theta_x, `u1`/`u2`=velocity. Fields were identified BY
MEASURED EXPONENT, not by assuming the schema.

## 3. `polar_radial_gate.py` — the radial machinery, and the formulation choice

`Psi = 0` at both beta edges, so `Psi = sum_k A_k(s) sin(2k beta)` satisfies both
angular conditions identically and the 2D Poisson problem decouples into 1D problems in
s — isolating exactly the untested direction. Solved on `s in [10,25]`
(`r = 2.2e4 .. 7.2e10`), NS=128, against the analytic per-mode solution:

| formulation | worst rel err over k=1..6 |
|---|---|
| RAW, solve `A_k` directly (spans **10.8 decades**) | 1.11e-14 |
| SUBST, `Psi = e^{(2+alpha)s} P`, solve `P_k` (constant) | **8.53e-16** |

**My conditioning worry was WRONG and is retracted**: RAW handles 10.8 decades fine —
a pure exponential is exactly what Chebyshev represents well. SUBST is 13x better and
is still the right default (it is this spec's own substituted line), but RAW is not
disqualified. This is the gate that proves the central claim of the log-polar move:
reaching `r ~ 1e8` costs a domain of length ~20 in s, where the box needed ~1e8.

## BASIS CHOICE — settled, and a near-miss worth recording

The g(beta) extracted from their profile has **g(wall) = 1.125 != 0** and
g(axis) -> 0. So:

| field | beta behaviour | basis |
|---|---|---|
| `Psi` | 0 at BOTH edges | Chebyshev + 2 taus (or sine, tau-free, in the 1D reduction) |
| `Om`, `B` | free at the wall | **Chebyshev — must NOT use a basis that vanishes at beta=0** |

Use Chebyshev in beta for ALL fields in the 2D problem (Dedalus wants shared bases
across an equation) and impose `Psi=0` at both edges by tau.

**NEAR-MISS, recorded so it is not repeated:** a sine expansion of g left a 62%
residual that decayed algebraically (`~K^-0.84`), which looks exactly like the
Abe-Ginsberg-Jeong edge singularity this document predicts. **It was not.** It was
`sin(2k beta)` forcing `g(0)=0` against a measured `g(0)=1.125`. Chebyshev in beta
drops the residual to 4.6e-3 at K=40. And that 4.6e-3 floor is most likely the noise
in MY extraction — 85 annulus-averaged bins off a 620^2 mesh — not a property of g.
**This data cannot decide the AGJ edge question either way; do not cite it as
evidence.** Deciding it needs a proper high-resolution angular extraction.

## 4. `polar_seed.py` — a VERIFIED initial guess (the actual blocker)

The blocker for Newton on a non-unique family was never the discretization, it was the
seed: the profile is a saddle, so Newton lands wherever it is pointed, including on the
trivial solution (which happened once here). Chen-Hou's converged profile is on disk, so
the seed comes from a solution known to exist.

**Key move:** do NOT interpolate the raw fields (10 decades). Interpolate the SCALED
fields `Ot = Om R^-alpha`, `Bt = B R^-(1+2alpha)`, which ARE the angular functions and so
are slowly varying; then reconstruct exactly. Their mesh is geometric in the far field
(measured spacing ratio a constant **1.0648**), i.e. **their mesh is already effectively
logarithmic in r** — independent corroboration of the log-polar frame.

| check | result |
|---|---|
| Ot, Bt s-independent over s in [19,30] | median 5.5e-4 / 1.3e-3, max 7.3e-3 / 1.0e-2 |
| exponent round-trip from reconstructed Om | -0.342398..-0.342412 vs -0.342400 (err ~1e-5) |
| angular consistency Ot(s=19.5) vs Ot(s=29.5) | 7.6e-3 |
| cross-gate vs angular_gate's annulus average, INTERIOR | **5.3e-3** |

### EDGE BEHAVIOUR — measured, and it settles the beta basis

At three well-separated s (20, 25, 30) the angular profile is identical to 3-4 digits
for every `eps = pi/2 - beta` down to `1e-8`:

    axis (beta -> pi/2):  Ot ~ eps^1.000 exactly, coefficient ~32.7   (linear zero)
    wall (beta -> 0):     Ot -> 1.12496, slope -> 0                   (flat, NONZERO)
    peak:                 Ot = 4.889 at beta = pi/2 - 0.03

So **both edges are REGULAR on this branch**: Om is smooth and odd at the symmetry line
with a simple linear zero, and flat at the wall. Consequences: impose `Om = 0` at
`beta = pi/2` (the solution meets it linearly — no weighted class needed HERE); never use
a beta basis that vanishes at `beta = 0`; the axis layer is ~0.03 rad wide, so Chebyshev
in beta — which clusters at BOTH ends — is the right basis.

**CAUTION on `angular_gate.py`'s EDGE bins.** Its annulus average is unreliable within
~0.05 rad of an edge, and the seed is right there. Polar binning of a CARTESIAN TENSOR
grid degenerates near an axis: the last beta bin at r in [1e10,1e12] collects points
whose absolute `y1` ranges from 0.004 to 7.8e9, and averaging those is not an estimator
of g. Compare interiors; trust the seed at the edges. (A naive all-points comparison
reports 24% and looks like a real disagreement; it is 5.3e-3 on the interior, with the
residue confined to ONE bin 0.004 rad from the axis.)

## 5. `polar_psi_gate.py` — the Psi seed AND the sign convention, vs their VELOCITY

`polar_ops_gate.py` only checked signs for internal consistency with this lab's
`u = skew(grad psi)`; it could not check them against DATA. And Psi is not stored, so its
seed has to be constructed — a constructed seed with a sign error is precisely what sends
Newton to zero. Both gaps close at once, because the far field predicts BOTH velocity
components from `Pt` alone:

    u_r * r^-(1+alpha) = -Pt'(beta),        u_b * r^-(1+alpha) = (2+alpha) Pt(beta)

with `Pt` from the angular ODE driven by the seed's `Ot`. Their `u1, u2` had been used
for nothing, so this is genuinely independent:

| comparison | rel L2 (interior, 368 of 400 pts) | sign |
|---|---|---|
| `u_r` vs `-Pt'` | **1.14e-3** | +1 |
| `u_b` vs `(2+alpha) Pt` | **5.31e-4** | +1 |

Magnitudes agree to 0.06%: `max|u_r|` 3.5297 vs 3.5317, `max|u_b|` 2.534 vs 2.5357.
Sign is **+1 and consistent across both components**, so **our convention matches
theirs** — no negation needed when seeding. This simultaneously re-confirms
`Psi ~ r^(2+alpha)`, the corrected `u_r`/`u_b` relations, and the angular ODE, from a
third direction.

### Validation chain now standing behind the solver

    operators      symbolic, exact        (caught a -1 sign error in this spec)
    far field      4/4 vs their data      ~1e-4
    radial         8.53e-16               over 10.8 decades
    angular ODE    1.65e-13
    seed Om, B     5.3e-3 interior        s-independent, exponent round-trip 1e-5
    seed Psi+sign  1.1e-3 / 5.3e-4        vs untouched velocity data, sign +1

Everything the solver needs is now gated EXCEPT (a) the double-Chebyshev tau construction
and (b) the method decision (Newton on the steady system vs time-marching the rescaled
equations — note `Perturbed_eqn/` contains `RK2_pertb_F.m` and `CFL_pertb.m`, which
suggests Chen-Hou MARCH rather than Newton; under investigation).

## 6. `polar_gauge_gate.py` — the GAUGE, verified against their stored constants

Two scaling symmetries means two normalizations, and pinning the wrong quantity leaves a
neutral direction — which here produced a singular Jacobian, a sign-flipping `c_w`, and
step norms of 1e6-1e7. The nonlinear gauge recorded above is testable rather than merely
citable, because their converged profile AND their converged `c_l, c_w` are both on disk.
Their mesh is UNIFORM near the origin (spacing 0.00390625), so one-sided 4th-order
corner differences are clean.

| formula | computed | stored | rel err |
|---|---|---|---|
| `c_l = 2 theta_xx(0) / omega_x(0)` | +3.00649823 | +3.00649798 | **8.4e-08** |
| `c_w = c_l/2 + u_x(0)` | -1.02942519 | -1.02942519 | **1.2e-09** |
| sign variants (control) | miss by 50% and 490% | | gate discriminates |

Measured corner values: `omega_x(0) = 1.19620314`, `theta_xx(0) = 1.79819132`,
`u_x(0) = -2.53267418`. **Use these two point conditions as the solver's gauge.**

### CORRECTION to section 2's field labels: `v` is NOT `theta_x`

Section 2 identified fields by measured EXPONENT, and `v`'s exponent is indeed that of a
first derivative of theta (-0.68471, against -0.68481 for `theta_x`) — which is why an
exponent-only identification accepted it. But `v/theta_x` varies from **0.50 to 2.84**
pointwise (non-constant, so not even a rescaling), and `v_x(0) = theta_xx(0)/2` exactly
rather than `theta_xx(0)`. **`v` = `theta / x1`.** [RESOLVED. `Build_profile_pertb_Nlev.m:125` is literally
`th = x1 .* v;`, and measurement confirms it exactly: `max|th - x1*v| = 0.0`
(not roundoff), against `max|v - d_1 th| = 0.295` for `max|v| = 0.668`. The exponent
test could never have distinguished them, because `theta/x1` and `theta_x` share the
exponent `2 alpha`. And it IS worth knowing: `v = theta/x1` is the variable Chen-Hou
actually EVOLVE, and in those variables the B-equation carries no velocity-gradient
terms at all (`F_pertb_2lev.m:139`) -- a real Jacobian simplification available to us.] The
identifications that matter are now pinned far harder than by exponent: this gate
reproduces BOTH scaling constants from `th` and `u1` to 8-9 digits, which is only possible
if `th = theta` and `u1` is the velocity component entering the gauge.

Lesson: an exponent match is a NECESSARY not a sufficient identification. Two distinct
fields that are both first derivatives of the same quantity share an exponent exactly.

### Axis convention, confirmed by physics rather than assumed

Reading `w[i,j]` as `(y1_i, y2_j)` with `beta = arctan2(y2, y1)` gives `Om` flat and
NONZERO as `beta -> 0` and `Om -> 0` linearly as `beta -> pi/2`. That is correct for a
WALL at `beta = 0` (vorticity need not vanish there) and a SYMMETRY AXIS at `beta = pi/2`
(omega is odd about it). The swapped reading would put a vanishing `Om` at the wall and a
nonzero one on the symmetry axis — physically wrong. So the convention is settled by the
data, not by an assumption about array order.

## 7. `polar_residual_gate.py` — the SUBSTITUTED SYSTEM, verified on their profile

"Verify the residual AT INIT before iterating" — done before the solver exists.

### The derivation (all prefactors cancel exactly)

With `Om = e^(a s) Ot`, `B = e^((1+2a)s) Bt`, `Psi = e^((2+a)s) Pt`, `a = c_w/c_l`, the
exponentials cancel because `c_l a = c_w` kills the `c_w Om` term and
`c_l(1+2a) = c_l + 2 c_w` kills the B source. Verified numerically:
`c_l a - c_w = 4.6e-09`, `c_l(1+2a) - (c_l+2c_w) = 9.2e-09`.

    c_l Ot_s + e^(a s)[(Pt_s+(2+a)Pt) Ot_b - Pt_b (Ot_s + a Ot)]
             = e^(a s)[cos b (Bt_s+(1+2a)Bt) - sin b Bt_b]
    c_l Bt_s + e^(a s)[(Pt_s+(2+a)Pt) Bt_b - Pt_b (Bt_s + (1+2a) Bt)] = 0
    Pt_ss + 2(2+a) Pt_s + (2+a)^2 Pt + Pt_bb = -Ot

`a < 0`, so `e^(a s)` DECAYS outward and the transport equations degenerate to
`c_l Ot_s = 0` — which is exactly the measured s-independence. Self-consistent.

### Psi comes from their VELOCITY, by quadrature

Psi is not stored, and getting it from the Poisson equation needs the very 2D machinery
under test (circular). Instead `u_r = -(1/r) Psi_b` with `Psi(wall) = 0` gives a 1D
angular quadrature per s: `Pt(s,b) = - INT_0^b u_r r^-(1+a) db'`. **Cross-check against
the angular-ODE Pt of section 5 (which never saw the velocity): rel L2 = 1.05e-3.**

### Results — all three equations, both regions

| window | R1 | R2 | R3 | note |
|---|---|---|---|---|
| s in [0,4] (INNER) | **6.6e-3** | **8.9e-3** | — | all terms O(1)-O(2.4): the DISCRIMINATING test |
| s in [20,30] (FAR) | **1.6e-2** | **9.9e-3** | **2.5e-2** | with a CUBIC seed |

### The interpolation floor — diagnosed, not guessed

With a LINEAR seed, far-field R1/R2 sat at ~1.0 and looked like a formulation error. Two
diagnostics separated the causes:

- **s-refinement**: R1 stayed flat at ~1.0 across NS = 20..320 (ds 0.53 -> 0.031). Grid
  noise amplified by `d/ds` would GROW; it did not. So not noise.
- **window sweep inward**: R1 fell monotonically 9.7e-1 -> 1.7e-1 -> 3.7e-2 -> 6.6e-3 as
  the window moved to s in [0,4], while the term scales rose from 9e-2 to 2.4.

Diagnosis: in the far field the equation demands `|c_l Ot_s| ~ 1.1e-2`, while LINEAR
interpolation's own spurious s-variation produces `4.7e-2` — error and signal the same
size, so the far-field balance is simply not resolvable from a linear seed.
Confirmed by fixing it:

| interpolant | median spurious s-spread in Ot | max spurious \|c_l Ot_s\| |
|---|---|---|
| linear | 5.6e-4 | 4.7e-2  (ABOVE the 1.1e-2 signal) |
| **cubic** | **1.5e-5** | **7.9e-3** (BELOW it) |
| quintic | 1.4e-5 | 7.9e-3  (= cubic, so cubic is converged) |

`polar_seed.py` now uses **cubic** (linear fallback). That alone improved far-field R1 by
**63x** and R2 by **97x**. **A seed's interpolation order is not a detail here — it sets
the floor on what the far field can say.**

## 8. BOUNDARY CONDITIONS — measured, and one of them CORRECTS this spec

All values below are s-INDEPENDENT (checked at s = 20, 25, 30 and agreeing to 3-4
digits), which is itself the far-field ansatz.

| edge | field | MEASURED behaviour | condition to impose |
|---|---|---|---|
| beta = 0 (WALL) | Om | flat, `Ot -> 1.12496`, slope 0 | none (free) |
| | B | flat, `Bt -> 3.88927` | none (free) |
| | Psi | `Pt -> 0` | `Psi = 0` |
| beta = pi/2 (AXIS) | Om | `Ot ~ eps^1.000`, coef ~32.7 | `Om = 0` (met linearly) |
| | B | **`Bt ~ eps^1.9992`** | **`B = 0`** (double zero) |
| | Psi | `Pt -> 0` | `Psi = 0` |

### CORRECTION: the spec's `d_b B = 0 (even)` at the axis is half right

Measured `Bt ~ eps^2` over four decades (`eps` 1e-2 -> 1e-5, `Bt` 0.04595 -> 4.9e-8;
ratio 9.4e5 against 1e6 predicted). So **BOTH `B = 0` AND `d_b B = 0`** hold there — a
DOUBLE zero, i.e. `B ~ y1^2` near the axis, not "even with `B != 0`". The spec recorded
only the derivative condition and omitted `B = 0`, which is the stronger and more useful
one. Note this is the SAME quadratic structure the spec already derives at the corner
(`(c_l + 2 c_w) B(0,0) = 0` with `c_l + 2 c_w = 0.9476 != 0`, hence `B ~ dist^2`) — now
measured along the whole symmetry axis. The two facts corroborate each other.

The B equation is FIRST order in beta, so impose `B = 0` and let `d_b B = 0` emerge.
Consistency check: on the axis `Pt = 0` and `Pt_s = 0`, so the B equation degenerates to
`c_l Bt_s - e^(a s) Pt_b (Bt_s + (1+2a) Bt) = 0`, which `Bt = 0` satisfies identically.

(Beware one-sided `np.gradient` at an endpoint where the field is ~1e-12: it returned a
spurious `-1.278` for `d_b Bt(pi/2)` and briefly looked like a real BC violation. A
geometric eps-sweep is the reliable measurement.)

### Characteristic direction fixes WHICH edge gets the transport conditions

`c_l = 3.006 > 0`, so the dominant operator `c_l d_s` sends characteristics OUTWARD in s.
The inflow edge for `Ot` and `Bt` is therefore `s = S0` (INNER), and they must get **no
condition at `s = S1`**. This is the same trap the spec already records from the box work
— "outflow BCs on a first-order transport equation gave a SINGULAR Jacobian, step norm
pinned at 1e12" — so the spec's own instruction to put three Robin conditions at `s = +S`
is WRONG for `Om` and `B`. Only `Psi` (second order in s) takes conditions at both ends.
In substituted variables the far-field condition on `Psi` is plain homogeneous Neumann
`d_s Pt = 0`.

## 9. BETA BASIS — a real trade-off, not a clean win

The tau nullity found in `polar_tau2d_gate.py` is exactly `order_s x order_b`, so driving
`order_b -> 0` by choosing beta bases that satisfy the conditions IDENTICALLY would remove
it entirely. The measured edge behaviour says which bases those are:

    Psi : 0 at both edges          -> sin(2k b)
    Om  : free at 0, 0 at pi/2     -> cos((2k+1) b)
    B   : free at 0, 0 at pi/2     -> also needs a zero at pi/2

Measured convergence of the least-squares fit to the seed (600 beta points, s=24):

| field / basis | K=4 | K=16 | K=64 | K=96 |
|---|---|---|---|---|
| `Pt` in `sin(2k b)` | 5.3e-3 | 2.2e-4 | 2.1e-6 | 6.4e-7 |
| `Ot` in `cos((2k+1) b)` | 5.3e-1 | 2.2e-1 | 1.5e-2 | 2.7e-3 |
| `Bt` in `cos(2k b)` | 6.6e-2 | 1.0e-2 | 2.3e-4 | 3.4e-5 |

**None converge exponentially, and `Ot` is bad.** Cause: `Ot` has a near-axis layer —
it rises from 1.125 at the wall to 4.889 at `beta = pi/2 - 0.03` then falls linearly to 0
— and a feature 0.03 rad wide needs `K >~ 52` before a trig basis even begins to resolve
it. Chebyshev CLUSTERS nodes at both ends and handles that layer far better (it reached
4.6e-3 at K=40 on noisier data).

So the choice is:

| option | tau nullity | axis layer |
|---|---|---|
| Chebyshev in beta (all fields) | `order_s x order_b`, cond ~1e17-1e19 | resolved well |
| mixed trig bases | **0** | resolved poorly for `Om` |

Per `polar_tau2d_gate.py` the null space does NOT contaminate the fields (null energy in
`Psi` ~1e-15; `Psi` agrees to 1.4e-14 across matsolvers while `tau` values differ by
O(1)), so **Chebyshev + accepted nullity is the leading choice**, with the open risk being
whether the NEWTON solve tolerates a rank-deficient Jacobian. Mixed-trig is the fallback
if it does not. A third option — resolve the layer by mesh grading instead of basis choice
— is what Chen-Hou's multi-level mesh appears to do; see the recon.

---

# 10. THE METHOD DECISION — SETTLED BY THE RECON: **MARCH, DO NOT NEWTON**

## Chen-Hou run no Newton anywhere

`run_pertb.m:1` — "Construct the approximate steady state by solving the dynamic
rescaling equations". The loop calls `RK_df` (RK2/Heun, `RK2_pertb_F.m:1-18`) and stops
at `run_pertb.m:68`: `l > 4501 || max(|Fv|,|Fw|) < 2e-10`. A recursive grep for
`newton|jacobian` over the whole `Perturbed_eqn/` folder matches ONE file, `map.m`, and
that is a mesh-map Jacobian. Verified against their saved 620x620 ASS: `max|Fw| =
1.86e-10`, `max|Fv| = 7.7e-12`, with `||w||_inf = 0.715` — a genuinely nonzero field.

## WHY marching works on what this spec called a saddle

This document asserted "the profile is a SADDLE ... Newton is correct precisely because
it does not care about stability". The saddle statement is true **of the steady system
with `c_l, c_w` as unknowns**. But Chen-Hou never treat them as unknowns:

    F_pertb_2lev.m:135-137     cl = 4 * vx1(1,1) / wx1(1,1);
                               cw = u1dx1(1,1) + cl / 2;

recomputed **at every RK stage**, so `c_l, c_w` are slaved algebraic functions of the
field, never independent unknowns, never integrals, never time-integrated. That
**removes the two unstable directions from the phase space** rather than stabilising
them, and what remains is an attractor — which is the literal content of their title,
"**Stable** nearly self-similar blowup".

## What makes the zero field inadmissible — the property we must reproduce

The two corner functionals are CONSERVED by the march: `rec[2] = 1.1962031505062485`
against the frozen `wbx0 = 1.196203150519860` (a difference of **1.4e-11**), and
`rec[3] = 0.8990955899969177` against `vbx0 = 0.899095589986449` (**1.0e-11**), after
~18787 outer steps. The amplitude is pinned by the INITIAL DATA through two conserved
corner functionals. That is the structural reason their march cannot collapse to zero —
the exact failure that already bit this lab.

## Their structural trick, worth copying

Two-tier substepping: one outer RK2 step WITH a Poisson solve, then **30 inner RK2 steps
with the velocity FROZEN** (`run_pertb.m:56-59`). Their own comment: "much faster (more
than 10 times) since it does not need to solve the Poisson equations." Transport is
relaxed ~31x more often than the elliptic coupling. No filtering during the march
(`LowPassFilter_wg` runs only in post-processing). Convergence is slow linear at the end
(e-folding ~4000 outer steps).

# 11. THE GAUGE SURVIVES LOG-POLAR — but it sets the INNER boundary

**The risk:** every scalar closure in both references is evaluated at `r = 0` exactly,
and log-polar deletes that point (`r = 0` is `s = -inf`). Section 6 only showed the
constants can be RECOVERED from stored data — not that they can be IMPOSED on a domain
excluding the corner.

**The diagnostic** (`polar_gauge_sweep.py`), basis-free so it does not depend on guessing
functionals. The two exact scaling symmetries have tangent vectors

    v_amp   = ( Om,   Psi,            2 B   )        [lam: c_l,c_w -> lam c_l, lam c_w]
    v_trans = ( Om_s, Psi_s - 2 Psi,  B_s - B )      [sig: c_l,c_w UNCHANGED]

and a gauge can determine two constants only if these are distinguishable on the domain.
In the pure power law `Om_s = a Om`, `Psi_s = (2+a)Psi`, `B_s = (1+2a)B`, so
`v_trans -> a * v_amp` **exactly parallel** — a dilation is indistinguishable from an
amplitude change for a pure power law. So the far field is structurally blind to the
gauge, and the question is how far in you must come.

| S0 | r_inner | angle | cond | seed data |
|---|---|---|---|---|
| -6 | 0.0025 | 50.8 deg | 4.4 | UNRESOLVED (0.6 cells) |
| -4 | 0.018 | 50.4 deg | 4.5 | 4.7 cells |
| **-2** | **0.135** | **48.7 deg** | **4.9** | 34.6 cells |
| 0 | 1 | 31.0 deg | 13 | 256 cells |
| +5 | 148 | 4.9 deg | 543 | |
| +20 | 4.9e8 | 0.18 deg | 3.9e5 | |

Measured decay `angle ~ e^(-0.224 s)` against the predicted `alpha = -0.342` — same
mechanism, confirming it is the `r^alpha` correction that carries the gauge signal.

**VERDICT: the gauge DOES survive, and the corner is NOT required.** The signal
SATURATES by `S0 ~ -2` (48.7 deg, already at the -4 value of 50.4 deg), where the seed
still has 34.6 cells of real data. **Put the inner edge at `S0 = -2` (`r = 0.135`).**
Placing it beyond `s ~ +5` is not a resolution compromise, it is a well-posedness
failure, with the symptoms already logged here: singular Jacobian, sign-flipping `c_w`,
step norms of 1e6-1e7, or silent convergence to zero.

**Recommended domain: `s in [-2, 25]`.** The outer end comes from the recon: `s_max ~ 37`
oversizes by ~12 units of s. `r ~ 1e8` is an accuracy-dependent crossover
(`|slope - alpha| ~ 0.0221 r^-0.401`), not an onset; Chen-Hou's own PRODUCTION mesh
reaches only `s = 30.7` — SMALLER than its 620-point predecessor at 35.7 — and their
far-field fit window is `s in [22.3, 29.0]`.

# 12. OTHER RECON CORRECTIONS TO THIS DOCUMENT

- **Tau nullity IS removable.** `valid_modes` masking of 4 tau columns + 4 equation rows
  takes cond from ~1e19 to **1.6e5**, drives `max|tau|` to roundoff, and retires the
  entire matsolver blacklist (all seven previously-failing matsolvers then succeed at
  1.6e-15). Masking is NOT needed for Psi accuracy (unmasked already reaches 1e-15..1e-13)
  but is worth doing for conditioning.
- **The "Newton trap" is a false alarm.** The tau VALUES are undetermined and O(1), but
  the tau STEP is not: an unmasked nonlinear NLBVP converged 5.06e-01 -> 4.80e-03 ->
  1.49e-04 -> 2.24e-12 -> 2.93e-15 with the tau value sitting at 0.96 throughout. At
  convergence the RHS vanishes and a fixed LU maps zero RHS to exactly zero step
  regardless of nullity. Do not distrust a converging step norm on these grounds.
- **DROP CGL.jl's C1 matching for the far field.** Its natural implementation needs the
  outer angular profiles slaved to each other via `T = C P^((1+2a)/(2+a))`, and that was
  tested and FALSIFIED on the reference profile: `T/P^kappa` varies by 2e10 across beta
  and the measured `d(lnT)/d(lnP) = -0.105` has the OPPOSITE SIGN to the predicted
  `+0.190`; near the axis `T ~ cos^1.94` while `P ~ cos^1.00`. Only the Psi angular
  Poisson relation is a clean leading-order relation — which is exactly why Chen-Hou
  SOLVE the Psi angular profile (`Profile_farfit.m:36`) but FIT G and T by nonlinear
  least squares (`FitWb.m:43`, `FitTheta.m:32`). **Solve for Psi's angular profile; FIT
  Om's and theta's.** Keep CGL.jl for gauge-by-scaling-symmetry and continuation.
- **`selfSimilarEulerBoussinesq` does not cover both systems.** Its `boussinesq/` path
  does not even parse (`def_fFull_*.m:9` is invalid MATLAB under a NOT IMPLEMENTED YET
  banner), has no driver, and no far-field treatment anywhere. Take only the 1D Burgers
  damped-Newton loop and the deflation operator.
- **Liu's thesis is oversold above.** No profile solver, no Newton, nothing in polar; Sec
  3.3-3.5 is a first-order-upwind forward-Euler dynamic-rescaling march on 40x40 / 80x80
  Cartesian reaching only `r ~ 6e3`, and he states outright he cannot extend the local
  profiles by solving the PDE. On his Sec 3.4 spectrum: the leading eigenvalue moves
  -0.587 -> -0.461 under his single refinement (the gap SHRINKS 21%), grid-independence
  is claimed only in 1D, and he limits it to "low frequency perturbations".
- **`solu.al` is NOT `|c_w/c_l|` live.** `solu.al = 0.3424000915844059` while
  `|solu.cw/solu.cl| = 0.34240009311696556` — different at the 9th digit, because
  `Build_profile_pertb_Nlev.m:188` sets `solu.al = abs(cwb/clb)` from the FROZEN
  historical values. Alpha in their pipeline is a hand-run scalar self-consistency loop
  across successive runs, converged to 1.5e-9. **There is no eigenvalue condition and no
  quantization anywhere in their method.**

---

# 13. THE MARCHER — `polar_march.py`

Built on the recon's verdict (section 10). Self-contained numpy/scipy: Chebyshev
collocation in both directions, boundary rows REPLACED rather than tau-lifted (which
sidesteps the double-Chebyshev tau nullity entirely — there are no tau unknowns to be
undetermined), a sparse Kronecker Poisson operator PREFACTORED once, and a hand-rolled
RK2/Heun matching `RK2_pertb_F.m`.

## alpha0 is FIXED in the change of variables

NOT slaved to `c_w/c_l`. If alpha tracked the moving `c_w/c_l` the substitution itself
would be time-dependent and would generate extra terms. Fixing `alpha0` keeps
`c_l, c_w` free scalars and every equation picks up ONE deviation factor
`(c_w - c_l alpha0)` which vanishes exactly at the fixed point:

    d_t Ot = (c_w - c_l a0) Ot - c_l Ot_s - e^(a0 s)[adv_O] + e^(a0 s)[src_O]
    d_t Bt = 2(c_w - c_l a0) Bt - c_l Bt_s - e^(a0 s)[adv_B]

## The gauge writes itself

`c_l` and `c_w` enter the RHS **affinely**, so `dF/dt` is affine in them for any linear
functional and pinning two is a 2x2 LINEAR solve — no iteration. The right two are the
system's own scaling tangents, in substituted variables:

    v_amp   = ( Ot,             2 Bt           )
    v_trans = ( Ot_s + a0 Ot,   Bt_s + 2 a0 Bt )

Requiring the evolution to have no component along either is the log-polar analogue of
Chen-Hou's corner slaving, and section 11 measured that this 2x2 is well-conditioned
precisely at `S0 = -2`.

## Gates (96x96, s in [-2,25])

| gate | result |
|---|---|
| G0 spectral derivatives | `d_s` 2.6e-13, `d_b` 1.3e-13 |
| G1 elliptic solve vs the seed's Psi (from their velocity) | 8.4e-3 |
| **G2 gauge RECOVERS Chen-Hou's constants** | **`c_l` 2.98036 (err 8.7e-3), `c_w` -1.02052 (err 8.7e-3), implied alpha -0.342414 vs -0.342400 (4e-5)** |
| G3 seed steadiness | `max|dOt|/|Ot|` 1.5e-2, `max|dBt|/|Bt|` 1.5e-3 |

**G2 is the one that matters.** It is the load-bearing claim of section 11 made good: a
gauge evaluated on a domain that EXCLUDES the corner recovers, to under 1%, the constants
Chen-Hou compute AT the corner — and the implied alpha to 4e-5.

## Short march (200 steps, dt 5e-4, 96x96): STABLE

`alpha` pinned at -0.342414 to six digits, `cond` flat at 562, `|Ot|max` unmoved at
4.8954, total drift from seed 1.55e-3. Cost 106 ms/step at 96x96 (sparse-LU fill-in
dominates; the operator is separable, so a Sylvester/eigendecomposition solve would be
far faster and is the obvious optimisation if long runs are needed).

## OPEN — the long march is NOT yet converging

At 64x64, dt 1e-3, the drift GROWS: `tau` 0 -> 0.4 -> 0.8 gives drift 1.5e-5 -> 6.3e-3
-> 1.4e-2 and `max|dOt|` 7.3e-2 -> 8.9e-2 -> 1.06e-1, with `c_l` creeping monotonically
up. `alpha` stays rock stable at -0.34241 throughout, which says the GAUGE is holding
and the growth is in the field, not the scalars.

Three candidate causes, in order of suspicion:

1. **Relaxation to a different discrete fixed point.** The seed is Chen-Hou's answer
   interpolated onto a different grid with a different inner boundary; the march should
   relax to THIS discretization's own fixed point, and that transient is not small.
   Diagnostic: drift must SATURATE. Ratio 6.3e-3 -> 1.4e-2 over `dtau = 0.4` is a factor
   2.25, i.e. an apparent rate ~2.0 — needs more `tau` to distinguish saturation from
   exponential growth.
2. **The frozen inner Dirichlet is a persistent forcing.** Holding `Ot, Bt` at seed
   values on `s = S0` while the interior relaxes injects a mismatch at the inflow edge
   every step. Fix: swap to the asymptotic inner Robin (`d_s Ot = (1-a0) Ot`, etc.,
   section 11), which lets the inner edge move.
3. **A genuinely unstable direction the gauge does not remove.** Chen-Hou's stability is
   stated on a RESTRICTED function space — perturbations must vanish QUADRATICALLY at the
   origin (`eq:normal_vanish`) — and this domain excludes the origin, so that restriction
   has no direct analogue here. If so the fix is a weighted projection, not a BC.

Distinguishing 1 from 3 is just a longer run; 2 is a one-line switch. Do both before
concluding anything about stability.

## 13b. MARCH RESULTS — honest status

### The grid-scale instability: found, and fixed

Unfiltered, the top-25% **beta** coefficient energy of `Ot` grew **7.1e-05 -> 3.2e-02
(449x)** over `tau = 6`, monotonically, while the **s** tail SATURATED at 2.3e-04. So the
growth was grid-scale and beta-directional. Cause: Chebyshev collocation of pure
advection carries no dissipation, and the beta-transport term `(Pt_s + mu Pt) Ot_b` is
exactly that. Both references have dissipation of their own (Chen-Hou B-splines + the
30-step frozen-velocity substepping; Liu FIRST-ORDER UPWIND); a bare collocation scheme
has none.

Fix: the Hou-Li exponential filter already validated in `dedalus_axisym.py:make_filter`.
Result: `tail_b` at `tau=6` falls **3.2e-02 -> 3.7e-03** and now SATURATES (~4.3e-03
through `tau=12`) instead of climbing.

**GOTCHA that cost a run:** a filter applied to the whole field OVERWRITES the Dirichlet
nodes the RHS pins every step, and the resulting inflow-edge mismatch feeds back. That
version DIVERGED by `tau = 2` (`max|dOt|` 0.073 -> 18.6) — strictly worse than no filter.
`filt()` now saves and restores the pinned rows.

### But the drift is NOT the filter's fault, and is NOT yet explained

Filtered vs unfiltered at `tau = 12` are nearly identical: `c_l` 3.200 vs 3.206,
`max|dOt|` 3.46e-01 vs 3.42e-01. So the grid-scale mode was real and is gone, and
something else drives the drift:

| tau | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| `max|dOt|` | 7.3e-2 | 1.1e-1 | 1.3e-1 | 1.9e-1 | 2.4e-1 | 2.5e-1 | 3.5e-1 |
| drift | 0 | 3.8e-2 | 5.9e-2 | 6.8e-2 | 7.3e-2 | 7.2e-2 | 8.4e-2 |
| `c_l` | 2.980 | 2.995 | 2.988 | 2.968 | 2.966 | 3.063 | 3.200 |

The DRIFT saturates (~7-8e-2) but the RESIDUAL grows ~4.7x and `c_l` wanders +-4% about
the reference 3.0065, rising monotonically in the last third. **This is not converging.**

**A METHODOLOGICAL NOTE ON MY OWN READING.** At `tau = 4`, on five points, the drift
increments were shrinking (6.2e-3, 4.8e-3, 4.0e-3, 2.8e-3, 2.2e-3) and I called it
"saturation, not instability". The full run refuted that: `c_l` turned around at
`tau ~ 2.4` and the excursion then grew ~9x. **Five points of a shrinking increment is
not convergence.** Run to the end before calling a trend.

### Cause 2 (frozen inner edge as persistent forcing) is REJECTED — informatively

Swapping the frozen seed-Dirichlet for the ASYMPTOTIC inner Robin (`d_s Ot = (1-a0) Ot`
etc., imposed by projecting the edge value from the interior) is CATASTROPHICALLY worse:
`c_l` collapses from 2.98 to ~0, drift reaches 29, residual 3e+02. The frozen edge is
STABILISING, not forcing.

Why: at `S0 = -2` the MEASURED log-slope is 1.272 while the asymptote is 1.342 (section
11's inward sweep). Imposing the exact asymptotic value forces a 5% inconsistency at an
INFLOW boundary of a transport equation, which is a strong forcing. The asymptotic Robin
is only legitimate where the asymptote actually holds, and at `s = -2` it does not yet.
If the Robin form is wanted, either push `S0` deeper (but the seed runs out of resolved
data below `s ~ -4`) or use the LOCALLY MEASURED slope rather than the asymptotic one.

### What remains

Cause 3 — a genuinely unstable direction the gauge does not remove. Note Chen-Hou's
stability is stated on a RESTRICTED function space (perturbations vanishing QUADRATICALLY
at the origin, `eq:normal_vanish`), and this domain EXCLUDES the origin, so that
restriction has no direct analogue here. If that is the cause, the fix is a weighted
projection, not a boundary condition.

The discriminator now running: does the drift SHRINK with N (48/64/96)? A discretization
artefact shrinks; a physical instability does not.

## 13c. THE DRIFT LOCALISED — and what it rules out

Four candidate causes tested; three eliminated.

**NOT grid-scale.** The Hou-Li filter cut `tail_b` 7.4x and made it saturate, but the
drift is unchanged (`tau=12`: `c_l` 3.200 filtered vs 3.206 unfiltered). Real instability,
now fixed, but not the cause.

**NOT the inner BC TYPE — and the Robin experiments were ill-posed.** Both the asymptotic
Robin AND the locally-measured-slope Robin (1.3062 / 1.6355, the seed's own values at
`s=-2`) blow up identically: `c_l` collapses to ~0, drift ~21-27, residual 1e+02-8e+02.
Since the slope value makes no difference, the problem is the CONDITION TYPE.
**`s = S0` is the INFLOW edge of a first-order transport equation, where the VALUE must
be prescribed, not a derivative relation** — a Robin there is ill-posed for `Ot, Bt`. The
Robin belongs to `Psi` alone (elliptic, second order). The BC structure already in the
code is correct; these runs were testing an ill-posed alternative.

**NOT resolution.** 48 / 64 / 96 across a 2x refinement:

| N | drift @tau=6 | growth tau 2->6 | `c_l` excursion |
|---|---|---|---|
| 48 | 6.41e-2 | 1.87x | 0.0401 |
| 64 | 6.76e-2 | 1.79x | 0.0383 |
| 96 | 6.99e-2 | 1.77x | 0.0362 |

Resolution-CONVERGED. So the drift is a property of the continuous problem as posed, not
a numerical artefact.

**NOT an inconsistent frozen inflow value.** That would put the residual at `s = S0`. It
is the opposite — the inner region has the SMALLEST residual.

### Where the residual actually lives

`max|dOt|` by band at `tau = 6` (64x64):

| s band | [-2,0) | [0,2) | [2,5) | [5,10) | [10,15) | [15,20) | [20,25) |
|---|---|---|---|---|---|---|---|
| `max|dOt|` | 1.8e-2 | 7.1e-2 | 1.6e-1 | **1.9e-1** | 1.5e-1 | 6.7e-2 | 1.3e-1 |
| nodes | 12 | 4 | 6 | 8 | 7 | 9 | 17 |

Concentrated in **`s in [2,15]`** — precisely the TRANSITION region where neither
asymptotic form holds (the inward sweep of section 11 measured the `Ot` log-slope falling
1.31 -> 0.59 -> 0.21 across `s = -2 -> 2`, and the far-field crossover is gradual,
`|slope - alpha| ~ 0.0221 r^-0.401`).

**And that is exactly where a single Chebyshev grid is WEAKEST.** Chebyshev clusters at
the ENDS: on `[-2,25]` at N=64 the mid-domain spacing is `ds ~ 0.67` while the edges get
~0.007 — a factor of ~100. The node counts above show it directly: 12 nodes in the first
2 units of `s`, but only 8 across `s in [5,10)`. The grid puts its fewest points where
the physics is richest, and refining N uniformly does not fix the RATIO.

**This is very likely why Chen-Hou's mesh is MULTI-LEVEL** rather than a single mapped
grid, and it is the next thing to change: split `s` into 2-3 Chebyshev panels (e.g.
`[-2,2]`, `[2,15]`, `[15,25]`) with C^1 matching, or use a coordinate map that clusters
in the transition band instead of at the endpoints. Uniform refinement is the one lever
already shown NOT to work.

## 13d. THE DRIFT IS A TRANSVERSE MODE, RATE ~0.35 — invariant to every numerical lever

### It is not a gauge failure

Decomposing the deviation from the seed into the orbit-tangent plane
`span(v_amp, v_trans)` and its complement:

| tau | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| total | 4.28e-1 | 8.44e-1 | 1.27e0 | 1.79e0 | 2.51e0 | 3.44e0 |
| along orbit | 0.1% | 0.3% | 0.9% | 2.1% | 4.2% | 7.2% |

**99.9% TRANSVERSE** early, still 93% at `tau = 6`. The gauge IS removing the orbit
directions; this is a mode it cannot and should not remove.

### Four numerical levers, no effect

| lever | range tested | drift @tau=6 |
|---|---|---|
| resolution | N = 48 / 64 / 96 | 6.41e-2 / 6.76e-2 / 6.99e-2 |
| Hou-Li filter | off / on | `c_l` 3.206 / 3.200 at tau=12 |
| node distribution (KTE) | kte = 0 / 0.9 / 0.99 | 6.76e-2 / 6.79e-2 / 6.81e-2 |
| inner truncation | S0 = -4 / -3 / -2 / -1 | rate 0.3459 / 0.3489 / 0.3505 / 0.3072 |

The KTE map did what it was built for — spacing ratio 40 -> 5.8, transition-band nodes
21 -> 28, min `ds` up 4.8x — and the drift did not care. **Growth rate ~0.35, intrinsic
to the posed problem.**

### CAUTION on section 11's conditioning number

The orbit-tangent angle in the gauge's OWN inner product (plain weighted L2, which is
what the 2x2 solve uses) is **8.13 deg, cond 555** — not the 48.7 deg / cond 4.9 that
`polar_gauge_sweep.py` reports. The sweep normalises each field component by its own RMS
first. The sweep's SHAPE against `S0` stands (that is what it was built to measure, and
the collapse to 0.18 deg at `S0=+20` is real), but its absolute number is not the gauge's
conditioning. Quote 555.

### Where this leaves things — and the right next move

A robust `+0.35` unstable transverse mode, against references that report stability
(Liu's leading eigenvalue -0.587 -> -0.461, though the recon notes he claims
grid-independence only in 1D and restricts to "low frequency perturbations", and
Chen-Hou's stability is stated on a space of perturbations vanishing QUADRATICALLY at the
ORIGIN, which a truncated domain cannot express). The discrepancy is UNRESOLVED and
should not be papered over.

Consistency note: the seed carries a ~1.5% steadiness residual (GATE 3) which is the
interpolation floor, and a mode at rate 0.35 amplifies that by `e^(0.35*6) = 8.2x` over
`tau = 6` — exactly the observed growth. So the march is behaving self-consistently; the
question is entirely whether the `+0.35` is real.

**Stop marching. Compute the SPECTRUM.** Linearise the gauged operator about the seed and
find its eigenvalues directly: it settles whether `+0.35` exists in one shot instead of
being inferred from a transient, it yields the EIGENFUNCTION (which will say what the
mode is and whether it is localised in the transition band where the residual sits), and
it is the actual scientific target anyway — the stability spectrum is what would settle
the axisym lattice question and Liu's `s >= 4` claim. The marcher already supplies
everything needed: `parts()` gives the affine RHS, so the Jacobian is a finite-difference
or matrix-free Arnoldi away.

# 14. THE SPECTRUM — `polar_spectrum.py`

Matrix-free Jacobian of the GAUGED RHS about the seed, central differences, then a DENSE
eigendecomposition. ARPACK was tried first and FAILED (4/14 eigenvectors after 4000
iterations) — exactly what a non-normal operator does to a Krylov method. Dense costs `n`
matvecs but is unconditionally reliable and returns the FULL spectrum, which for a
non-normal operator is the point. Measured departure from normality
`||JJ^T - J^TJ|| / ||J||^2 = 0.10-0.12`.

## Spurious modes separate cleanly from physical ones by RESOLUTION SCALING

| N | max \|Im\| among leading | unstable count |
|---|---|---|
| 24 | 24.6 | 23 |
| 32 | 35.8 | 39 |
| 40 | 47.2 | 59 |

Both scale ~linearly with N. Frequency `~ c_l/ds ~ N` and a count growing with N are the
signatures of GRID-SCALE modes. **Discard everything with large `|Im|`.** (These are the
same modes the Hou-Li filter suppresses in the march, which is why the filter helped the
tail without touching the drift.)

## One mode CONVERGES — and it is the one the march sees

| N | eigenvalue | period |
|---|---|---|
| 24 | +0.2521 +- 0.5973i | 10.52 |
| 32 | +0.1925 +- 0.6324i | 9.94 |
| 40 | +0.2036 +- 0.6401i | 9.82 |

Frequency converging cleanly (0.597 -> 0.632 -> 0.640); growth rate in the band
0.19-0.25. **Its period ~9.8 matches the `c_l` oscillation measured in the march**
(`c_l` turned at `tau ~ 2.4`, minimum at `tau ~ 6.8`, i.e. half-period ~4.4). The march's
apparent rate ~0.35 exceeding the eigenvalue ~0.20 is consistent with non-normal
transient amplification on top of the modal growth.

**So the instability is real, oscillatory, and converged — for the problem AS POSED HERE.**

## The discrepancy with the references, and the leading hypothesis

Chen-Hou and Liu both report stability. All the gates on the equations pass (the
substituted system verifies on their converged profile to 0.7-2.5%; the gauge recovers
their `c_l, c_w` to 0.87% and `alpha` to 4e-5), so an outright equation error is
unlikely though not excluded.

**The concrete remaining difference is the GAUGE.** Theirs pins two POINT functionals at
the corner and thereby CONSERVES `omega_x(0)` and `(theta/x)_x(0)` — measured conserved
to **1e-11 over ~18787 steps**. The recon flagged those conserved corner functionals as
"the structural reason their march cannot collapse to zero — the property we must
reproduce". **This implementation does NOT reproduce it.** An L2 projection removes the
two orbit directions in a mean-square sense but conserves nothing pointwise, and the
corner where their functionals live is exactly the point log-polar deletes.

That is a specific, testable difference and it is the next thing to try: replace the L2
projection with two functionals that are pointwise-conserved analogues evaluated at the
INNER EDGE `s = S0` (where section 11 measured the gauge signal is still strong) rather
than at `r = 0`, and re-run this spectrum. If the unstable pair moves to the left half
plane, the gauge was the whole story.

Do NOT conclude from the present +0.20 that the Luo-Hou profile is unstable. The claim
supported by this work is narrower and should be stated as: *the log-polar formulation on
`s in [-2,25]` with an L2-projection gauge has a converged unstable oscillatory mode at
`Re ~ +0.20`, period ~9.8.* Whether that is a property of the profile or of this gauge is
open, and the experiment above distinguishes them.

# 15. THE ANSWER — the unstable mode is real, and it indicts the FRAME, not the gauge

`polar_mode_battery.py`, N=28, three tests in one pass.

## A. It is NOT a boundary artefact

| s band | [S0,0) | [0,2) | [2,5) | [5,10) | [10,15) | [15,20) | [20,S1) |
|---|---|---|---|---|---|---|---|
| energy fraction | 0.010 | 0.037 | 0.200 | **0.289** | 0.171 | 0.124 | 0.169 |

Amplitude at the inner edge / peak = **0.0000**. Peak at **s = +6.15**, beta-profile
peaking at the **WALL** (`beta/(pi/2) = 0.014`). **66% of the energy sits in
`s in [2,15]`** — the same transition band where the march's residual concentrates
(section 13c). So it is a transition-band, wall-hugging structure, not something the
inner boundary manufactured.

## B. It is NOT the gauge — section 14's hypothesis is DEAD

| gauge | leading low-\|Im\| unstable | period |
|---|---|---|
| L2 projection (default) | +0.210 +0.686i | 9.16 |
| **FROZEN c_l,c_w (NO GAUGE AT ALL)** | **+0.126 +0.918i** | 6.85 |
| point conditions at the inner edge | +0.480 +0.960i | 6.54 |
| inner-weighted projection | +0.221 +0.935i | 6.72 |

With `c_l, c_w` frozen at the reference constants the operator carries no gauge
whatsoever, and it is STILL UNSTABLE. **Section 14 proposed that reproducing Chen-Hou's
conserved corner functionals would move the mode into the left half plane. That is
refuted — retract it.** The rate does depend on the gauge (0.126 .. 0.480), which is
consistent with the note there that s-translation is NOT an exact symmetry of a truncated
domain, so no gauge here is a clean slice; but none of them stabilises.

## C. It depends strongly on the INNER TRUNCATION

| S0 | -4 | -3 | -2 | -1 |
|---|---|---|---|---|
| Re | +0.427 | +0.127 | +0.210 | +0.054 |
| period | 19.0 | 16.6 | 9.16 | 7.78 |

**CAVEAT, stated plainly:** the periods differ so much across `S0` that the selector is
probably tracking DIFFERENT modes, not one mode moving. The defensible reading is only
"the low-`|Im|` unstable spectrum changes substantially with `S0`", not a clean trend.
Tracking a single mode would need eigenvector continuation, which has not been done.

## WHAT THIS MEANS

The mode is a genuine mode of the problem AS POSED — not numerical (converged in N, and
the grid-scale modes are separately identified by their `|Im| ~ N` scaling), not the
gauge (survives with none), not the inner boundary (eigenfunction vanishes there). But
the spectrum is strongly sensitive to WHERE the inner cut is placed, so it cannot be
identified with a mode of the untruncated profile problem.

**The conclusion is about the FRAME.** Stability of this profile is decided in a
neighbourhood of the corner — Chen-Hou's gauge lives at `r = 0`, their conserved
functionals are corner derivatives, and their stability is stated for perturbations
vanishing QUADRATICALLY AT THE ORIGIN. **Log-polar structurally deletes `r = 0`
(`s = -infinity`), so it cannot express that function space at all, at any `S0`.**

So: **log-polar is the right frame for the FAR FIELD and the profile's outer structure —
which is exactly what sections 1-9 established and gated to 1e-3..1e-15 — and it is the
WRONG frame for the STABILITY problem.** That is not a defect in the implementation; it
is a property of the coordinate change. It also explains, in one stroke, why Chen-Hou
work on a CARTESIAN tensor mesh that INCLUDES the origin and is only geometric (i.e.
log-radial) in the TAIL: they need the corner for the gauge and for the spectrum, and the
log-radial grading only where the power law lives. The mesh recon measured exactly that
structure — uniform spacing 0.00390625 near the origin, then a machine-exact ratio 1.15
over the last 127 intervals.

## THE ROUTE THAT FOLLOWS

Do NOT keep pushing `S0` inward — the seed runs out of resolved data below `s ~ -4`
(4.7 cells) and no finite `S0` recovers the corner. Two options:

1. **Hybrid domain** — a small Cartesian (or plain polar, `r` not `ln r`) patch covering
   `r in [0, r_m]` including the corner, matched to the log-polar solve outside at
   `s_m = ln r_m`. All the far-field machinery in sections 1-9 is reusable unchanged; only
   the inner patch is new, and it is small and smooth (the fields are analytic at the
   corner: `Om ~ w_x(0) y1`, `B ~ th_xx(0) y1^2 / 2`).
2. **Do what Chen-Hou do** — a single Cartesian tensor mesh, uniform near the origin and
   geometric in the tail. Loses log-polar's clean far-field Robin conditions, but the
   far-field treatment is already solved and gated here and could be carried over as the
   outer fit.

Option 1 keeps everything already verified and adds the one piece that is missing. It is
the recommended route.

# 16. THE CORNER-INCLUSIVE FRAME `xi = ln(1+r)` — built, two bugs fixed, gauge still open

## The map

`xi = ln(1+r)`: `xi ~ r` at the corner (so **r = 0 is at xi = 0, a finite INCLUDED
point**) and `xi ~ ln r` far out (so every far-field result in sections 1-9 carries over).
The substitution `(1+r)^p = e^(p xi)` is EXACTLY the old `e^(p s)` in the far field. With
`g = r/(1+r) = 1 - e^(-xi)`:

    r d_r = g d_xi ,   d_1 = (1/r)(g cos b d_xi - sin b d_b)
    Lap   = (1/r^2)(g^2 d_xixi + g(1-g) d_xi + d_bb)

    g^2(Pt_xixi + 2 mu Pt_xi + mu^2 Pt) + g(1-g)(Pt_xi + mu Pt) + Pt_bb = -g^2 Ot

**No singular coefficients**, and at `g -> 0` the Poisson equation degenerates to
`Pt_bb = 0`, which with `Pt = 0` on both beta edges forces `Pt = 0` at the corner: the
regularity condition at the origin APPEARS ON ITS OWN instead of being imposed. That is
the structure log-polar could not express at any `S0`.

## Two real bugs found and fixed

1. **The dilation generator is `g d_xi`, not `d_xi`.** With `xi = ln(1+r)`, `r -> lam r`
   gives `d xi / d sigma = r/(1+r) = g`, so dilation is NOT a translation in `xi`. The
   correct scaling tangents are

       d Ot / d sigma = g (Ot_xi + a Ot)
       d Bt / d sigma = g (Bt_xi + (1+2a) Bt) - Bt

   both reducing to the log-polar forms at `g = 1`.

2. **Do not zero the axis beta-column of the seed.** The beta grid stops at `pi/2 - 1e-3`,
   NOT at `pi/2`, so the last column is NOT zero (measured `Ot = 0.326` there). Zeroing it
   put a jump in the seed and corrupted the beta derivative — `Ot_b` came out 0.489
   against the correct 0.702, a 30% error that propagated into the advection bracket.
   `polar_march.py` zeroes the TIME DERIVATIVE on that column (freezing the value); it
   never zeroes the value. Only the CORNER ROW is genuinely zero (`r = 0` so `y1 = 0`).

## Verification: the frame reduces to log-polar in the far field, term by term

At matched physical `r`, `xi`-frame vs log-polar (64x64 both):

| r | Pt | adv | src | Ot0 | KO | LO | MO |
|---|---|---|---|---|---|---|---|
| 1e6 | 1.51524 / 1.51519 | 1.76897 / 1.76905 | 1.81147 / 1.81163 | agree 6 dig | 6.911e-4 / 6.926e-4 | 4.7697e-1 / 4.7696e-1 | 1.3933 / 1.3933 |
| 1e10 | 1.51587 / 1.51587 | 1.77221 / 1.77174 | 1.81048 / 1.81037 | agree 6 dig | — | agree | agree |

**Every piece agrees.** The equations are right.

## STILL OPEN: the gauge returns the wrong answer

| | c_l | c_w | alpha |
|---|---|---|---|
| reference | 3.00650 | -1.02943 | -0.34240 |
| log-polar (node sum) | 2.98043 | -1.02054 | -0.34241 |
| log-polar (Clenshaw-Curtis) | 2.98301 | -1.02142 | -0.34241 |
| **corner (node sum)** | **4.65520** | **+0.48882** | **+0.10500** |
| **corner (Clenshaw-Curtis)** | **4.39026** | **+0.50797** | **+0.11570** |

`c_w` has the WRONG SIGN. Ruled out so far:
- NOT resolution: `c_l` = 4.655 / 4.653 / 4.651 at `Nx` = 64 / 96 / 144. Converged to a
  wrong answer.
- NOT the quadrature weighting: Clenshaw-Curtis moves it 4.66 -> 4.39, nowhere near 2.98,
  while log-polar is unchanged under the same switch.
- NOT the equations, the seed, or the tangents: every far-field piece matches log-polar
  to 4-6 digits (table above), and both bugs above are fixed.
- Every restricted `xi` window gives `alpha` in +0.10..+0.15, so it is systematic, not
  localised to one band.

**Leading hypothesis (untested):** the discrepancy must come from the region the `xi`
frame ADDS and log-polar never had, `r in [0, 0.135]`. There `1/g` is largest and the
REFERENCE SEED IS WORST — Chen-Hou's mesh spacing near the origin is 0.00390625, and the
Chebyshev `xi` nodes reach down to `xi ~ 0.0156` (`r ~ 0.0157`), i.e. FOUR cells of
reference data. Interpolated garbage there, amplified by `1/g`, would poison a global
inner product exactly as observed.

**Next step, well defined:** replace the interpolated seed for `r < ~0.05` with the
ANALYTIC corner form, using constants already measured to 8-9 digits in
`polar_gauge_gate.py`:

    Om  = w_x(0) y1                 = 1.19620314 * r cos b
    B   = th_xx(0)/2 * y1^2         = 0.89909566 * r^2 cos^2 b
    Psi from u_x(0) = -2.53267418

blended smoothly into the interpolated seed above `r ~ 0.1`. That removes the unreliable
near-corner interpolation entirely rather than trying to weight around it.

# 17. CORNER FRAME WORKING — one missing term was the whole problem

## The bug: `LB` dropped the `c_l` contribution from the RHS

The Bt equation's right-hand side is `(c_l + 2 c_w) B`, so **`c_l` appears there too** and
that contribution survives the substitution:

    LB = -g (Bt_xi + (1+2a) Bt) + Bt          <-- the `+ Bt` was missing

At `g = 1` this reduces to `-(Bt_xi + 2 a Bt)`, i.e. exactly `polar_march.py`'s `LB`.
`LO` is unaffected because the Om equation's RHS is `c_w Om`, with no `c_l`.

**How it was found:** printing the gauge 2x2 in both frames on the SAME physical window
showed `LB` at `r ~ 1e9` as `-1.0049` (corner) against `+2.1800` (log-polar) -- a
difference of exactly `-3.1849 = -Bt`. Everything else (`KO, LO, MO, KB, MB, vT`, the
seed, the fields) already matched to 4-6 digits, which is what made a single-term
discrepancy findable at all.

## Gates now match log-polar, WITH the corner in the domain

| gate | corner frame | log-polar |
|---|---|---|
| G0 spectral `d_xi` | 2.6e-13 | 2.6e-13 |
| G1 Poisson vs the seed's Psi | 8.3e-3 | 8.4e-3 |
| **G2 gauge `c_l`** | **2.980941 (0.85%)** | 2.980359 (0.87%) |
| **G2 gauge `c_w`** | **-1.020718 (0.85%)** | -1.020517 (0.87%) |
| **G2 implied alpha** | **-0.342415** | -0.342414 |
| G3 seed steadiness | 1.494e-2 / 1.493e-3 | 1.538e-2 / 1.481e-3 |

## Three dead ends recorded, because each cost a cycle

- **Not the near-corner seed.** An ANALYTIC corner seed (`Om = 1.19620314 r cos b`,
  `B = 0.89909566 r^2 cos^2 b`, blended in log r over [0.02, 0.1]) changed `c_l` by 1e-5.
  Worth keeping anyway: it agrees with the interpolated seed to **0.17-0.5% over
  r in [0.01, 0.05]**, which independently confirms both corner constants, and it removes
  a region where the reference mesh has only ~4 cells.
- **Not the quadrature weighting.** Clenshaw-Curtis moved the corner answer 4.66 -> 4.39
  and left log-polar unchanged at 2.98.
- **Not resolution.** `c_l` = 4.655 / 4.653 / 4.651 at `Nx` = 64 / 96 / 144.

**The lesson:** when two formulations should agree and do not, compare them TERM BY TERM
on matched physical points and descend to the smallest disagreeing quantity. Four global
hypotheses (seed, quadrature, resolution, domain) were all wrong; the answer was one
missing algebraic term, visible immediately once `LB` was printed side by side.

# 18. SPECTRUM WITH THE CORNER INCLUDED — and a methodological problem

| N | leading low-\|Im\| | period |
|---|---|---|
| 24 | +0.50390 +- 0.12736i | 49.3 |
| 28 | +0.57486 +- 0.23918i | 26.3 |
| 32 | +0.51657 +- 0.26477i | 23.7 |

The log-polar mode (`+0.20 +- 0.64i`) is GONE, but a different unstable mode appears, and
**it is NOT converged in N** (`Im` drifting 0.127 -> 0.239 -> 0.265). So the section 15
hypothesis is NOT confirmed: including the corner did not stabilise the operator.

## The real problem, and it invalidates every spectrum computed so far

**A Jacobian spectrum is a STABILITY spectrum only AT A FIXED POINT.** The seed carries a
~1.5% steadiness residual (G3, in BOTH frames), so every eigenvalue reported in sections
14, 15 and above is the spectrum of the linearisation about a point that is NOT a steady
state, and is contaminated by that offset. Chen-Hou march to `max|F| < 2e-10` FIRST and
only then linearise -- that ordering is not incidental.

This also re-frames the earlier "unstable with the gauge FROZEN" result (section 15,
Test B): that too was computed about a non-fixed-point, so it does not establish what it
was taken to establish.

**Required order, from here:**
1. march the corner frame to a genuinely converged profile (residual down by orders of
   magnitude, not percent),
2. THEN linearise about the converged state,
3. THEN read the spectrum.

Step 1 is now plausible for the first time, because the corner frame is the first
formulation here whose gauge reproduces the reference constants AND whose domain contains
the point where the gauge is defined.

# 19. CHEN-HOU'S POINT GAUGE, AND WHERE THIS STANDS

## Their gauge is now computable, and it works

Only possible because the corner is IN the domain. In `xi` variables at `xi = 0`
(`xi ~ r`, `e^(p xi) -> 1`):

    w_x(0) = Ot_xi(0, b=0)      th_xx(0) = Bt_xixi(0, b=0)
    u_x(0) = -d_b[ Pt_xixi(0,b)/2 ]|_{b=0}       (Psi ~ r^2 h(b), u_x(0) = -h'(0))

| N | w_x(0) | th_xx(0) | u_x(0) | c_l | c_w |
|---|---|---|---|---|---|
| 48 | 1.196426 | 1.775419 | -2.494318 | +2.96787 | -1.010383 |
| 64 | 1.194938 | 1.813060 | -2.492232 | +3.03457 | -0.974949 |
| 96 | 1.196583 | 1.776163 | -2.486123 | +2.96873 | -1.001761 |
| ref | 1.19620314 | 1.79819132 | -2.53267418 | +3.006498 | -1.029425 |

`w_x(0)` to 0.02-0.1%; `c_l` to ~1%. The mixed third derivative for `u_x(0)` is the noisy
one (1.5-1.9%) and does not converge monotonically.

## But neither gauge converges the march

| gauge | behaviour over tau = 0..4 (48x48, filter on) |
|---|---|
| L2 projection | residual 6.85e-2 -> 1.14e-1, monotone growth; `w_x(0)` drifts **98%**, `th_xx(0)` **96%** |
| conserving (freeze the two corner functionals) | residual oscillates 2.6e-1 -> 8.7e-1 -> 1.3e-1 -> 6.5e-1; `w_x(0)` drift 4.5e-2, `th_xx(0)` 6.3e-1 |

Two things worth separating:

- **The L2 gauge does NOT reproduce Chen-Hou's conservation property.** Their `w_x(0)` and
  `(theta/x)_x(0)` hold to 1e-11 over ~18787 steps; under the L2 gauge they move by 98%.
  This confirms, rather than refutes, the concern recorded in section 14.
- **The conserving gauge holds `w_x(0)` far better (4.5e-2 vs 98%) but still does not
  conserve it**, because imposing `d/dt = 0` at each RK stage is undone by the RK2 update,
  the Hou-Li filter, and the boundary pinning, none of which preserve a point functional.
  A genuinely conserving scheme would have to RE-PROJECT the functionals after each full
  step, not merely annihilate their time derivative within it.

## WHERE THIS STANDS — stated plainly

ESTABLISHED this session:
- The corner-inclusive frame `xi = ln(1+r)` is correct and gated: it reproduces every
  log-polar gate (`c_l` 0.85%, `alpha` to 1.5e-5, seed steadiness 1.49e-2) WITH `r = 0`
  inside the domain, and the origin regularity condition emerges from the degenerate
  Poisson equation instead of being imposed.
- One genuine algebra bug found and fixed (`LB` missing the `c_l` contribution from the
  RHS `(c_l + 2 c_w) B`), found by term-by-term comparison against log-polar after four
  global hypotheses had all been wrong.
- Chen-Hou's point gauge is implementable here and recovers their constants to ~1%.

NOT ESTABLISHED, and not to be claimed:
- **No converged profile.** Every march here grows; Chen-Hou reach `2e-10`. Until a
  profile converges, nothing can be said about stability.
- **Every spectrum computed in sections 14, 15 and 18 is invalid as a STABILITY
  spectrum**, because each was taken about a seed carrying a ~1.5% residual. A Jacobian
  spectrum at a non-fixed-point is not a stability spectrum. This retroactively weakens
  section 15's "not the gauge" conclusion, which was drawn from exactly such a spectrum.

REMAINING UNTRIED CANDIDATE, and it is a real algorithmic difference:
**their two-tier substepping** -- one outer RK2 step WITH the Poisson solve, then **30
inner RK2 steps with the velocity FROZEN** (`run_pertb.m:56-59`). That is not a speed
trick only; it relaxes transport 31x more often than the elliptic coupling, which changes
the effective dynamics of the coupled system. Every march here does one Poisson solve per
stage. That is the next thing to try before concluding anything about stability.

# 20. CONVERGED PROFILE — Newton, after the march was ruled out on evidence

## Why marching could never work HERE (and why that is not a contradiction)

The recon is right that Chen-Hou MARCH. But the measurement that decides what WE should
do is the seed residual against resolution, in the corner frame:

    N =      32       48       64       96      128
          1.12e-2  1.41e-2  1.49e-2  1.54e-2  1.55e-2      -- FLAT

Flat in N means it is not discretization error (that falls spectrally on a smooth
profile) -- it is the seed's INTERPOLATION FLOOR, concentrated at `xi in [2,10]`
(`r in [6, 2.2e4]`), the transition region. Chen-Hou march from THEIR OWN discrete
representation; we march from an interpolation onto a different grid carrying 1.5% error
in the most active region, and any growing direction amplifies it. That is exactly what
every march did: two frames, three gauges, filter on/off, N = 24..144, KTE redistribution,
inner truncation -4..0, one- and two-tier substepping. **Two-tier changed nothing** --
matched at equal effective tau it integrates the same trajectory ~31x cheaper, so the
"local vs global relaxation" hypothesis for it was wrong.

Newton does not care whether the fixed point is stable. POLAR_SPEC line 119 said exactly
that and was right; only the inference "therefore Chen-Hou use Newton" was wrong.

## The formulation that works: c_l, c_w as UNKNOWNS + two corner constraints

First attempt let the gauge compute `c_l, c_w` FROM the field. Newton converged
quadratically (7.5e-3 -> 8.0e-7) but slid along the free scaling family: `c_l = 1.513`
against 3.0065, field 28-45% smaller, `alpha` off by 4.6%. **The scaling directions must
be removed BY CONSTRAINT, not measured after the fact** -- which is why Chen-Hou pin the
two corner functionals. Promoting `c_l, c_w` to unknowns and adding

    Ot_xi(0, b=0)   = w_x(0)   = 1.19620314
    Bt_xixi(0, b=0) = th_xx(0) = 1.79819132

gives an (2N+2)-square system. Result at N=28: `||F||` **2.14e-2 -> 1.35e-11**, ratios
0.106 / 0.194 / 0.0038 / 0.0000 -- quadratic. The field GREW (|Ot| 5.08 vs seed 4.68), so
this is not the converge-to-zero trap.

## The outer alpha loop closes

`alpha0` is fixed inside the substitution, so it must be made self-consistent with
`c_w/c_l` -- the recon noted Chen-Hou run exactly this as a hand loop across successive
runs. At N=36:

| iter | alpha0 | alpha_out | gap | \|\|F\|\| |
|---|---|---|---|---|
| 0 | seed | -0.342124 | 2.76e-4 | 2.21e-11 |
| 1 | -0.342124 | -0.342109 | 1.48e-5 | 4.68e-12 |
| 2 | -0.342109 | -0.342108 | **7.95e-7** | **2.09e-12** |

**CONVERGED SELF-SIMILAR PROFILE**, `||F|| = 2.1e-12` (Chen-Hou stop at 2e-10), with

    alpha = -0.342108   against their -0.342400      -- 0.085%
    c_l   =  3.063185   against  3.006498            -- 1.9%
    c_w   = -1.047940   against -1.029425            -- 1.8%

`alpha` -- the physically meaningful eigenvalue -- is 20x more accurate than either `c`
alone, as expected since the `c`s carry the amplitude normalisation and `alpha` is their
ratio. Resolution is not yet monotone (`c_l` = 3.281 / 3.063 / 3.118 at N = 28/36/44), so
these are converged-in-residual but not yet converged-in-N.

# 21. THE FIRST LEGITIMATE STABILITY SPECTRUM

Taken at the Newton fixed point (`||F|| ~ 1e-11`), NOT at the seed. The gauged operator is
the SCHUR COMPLEMENT of the Newton Jacobian, `A - B D^-1 C`, i.e. the field dynamics with
`c_l, c_w` slaved to hold the two corner constraints -- which IS Chen-Hou's closure.

| N | leading REAL | second |
|---|---|---|
| 28 | +1.12459 | +0.59886 +- 0.22759i |
| 36 | **+1.04786** | +0.50918 |
| 44 | **+1.05340** | +0.42975 |

## The leading eigenvalue is 1, and that is a SYMMETRY, not an instability

Self-similar blowup carries a TIME-TRANSLATION freedom: the blowup time `T` is a free
parameter, and in rescaled time `tau = -ln(T-t)` shifting `T` generates a mode growing
exactly like `e^tau`, i.e. **eigenvalue exactly 1**. It must be quotiented out alongside
the two scaling modes. Measuring **+1.048 / +1.053** against a theoretical 1 is therefore
a ~5% CHECK ON THE WHOLE CONSTRUCTION -- the frame, the gauge, the constraints and the
Schur complement all have to be right for it to land there.

## What is still open

The next eigenvalue (+0.509 at N=36, +0.430 at N=44) is NOT converged in N and is
positive. Liu reports his leading eigenvalue as -0.587 -> -0.461. The magnitudes are
suspiciously close to mine with the opposite sign, which raises a SIGN-CONVENTION
question (decay-rate-positive versus growth-rate-positive) that must be settled before any
comparison is made -- but note the time-shift mode must be `+1` in a growth convention, so
the conventions cannot simply be mirrored. Do not compare to Liu until this is resolved.

Next: quotient out the time-shift mode explicitly (as the two scaling modes already are),
then re-read the spectrum; and push resolution until `c_l` and the second eigenvalue
converge in N.

# 22. CORRECTION to section 21, and the mode IDENTIFIED

## The section 21 spectrum was the UNCONSTRAINED operator

The constraints `g1 = Ot_xi(0,b=0) - w_x(0)` and `g2 = Bt_xixi(0,b=0) - th_xx(0)` depend
ONLY ON THE FIELD, so `D = dg/dc = 0` **exactly** (measured `||D|| = 0.00e+00`). The Schur
complement `A - B D^-1 C` is therefore undefined, and section 21's code had a
`try/except` that silently fell back to `Sc = A`. **Everything reported in section 21 is
the spectrum of the UNCONSTRAINED field operator `A`, not the gauged one.** Stated as
"the first legitimate stability spectrum" -- it was legitimate in being taken at a genuine
fixed point, which was the point at issue, but it was not the constrained operator.

**A `try/except` around a linear solve hid a structural fact.** If the fallback had been
loud, the `D = 0` structure would have been obvious immediately.

## The correct constrained operator

Differentiate the constraints in time rather than inverting `D`:

    d/dt g = Cg (A x + B c) = 0   =>   c = -(Cg B)^-1 Cg A x
    L = [ I - B (Cg B)^-1 Cg ] A = P A          (a PROJECTION, well conditioned:
                                                 cond(Cg B) = 29.4 / 6.58 / 9.09)

`P B = 0` and `Cg P = 0`, so `range(P)` is the constraint tangent space and `L` has
exactly 2 zero eigenvalues by construction -- both observed.

## The +1.05 mode IS a symmetry direction — measured, with a control

Overlap of the leading eigenvector of `A` with `span{v_amp, v_trans}`:

| N | leading eig | overlap with vA | with vT | **with the SPAN** | random control |
|---|---|---|---|---|---|
| 28 | +1.12459 | 0.261 | 0.134 | **0.993** | 0.027 |
| 36 | +1.04786 | 0.267 | 0.138 | **0.999** | 0.022 |
| 44 | +1.05340 | 0.253 | 0.141 | **1.000** | 0.029 |

Essentially ENTIRELY in the symmetry span, at three resolutions, against a random vector
that lands at ~3%. The individual overlaps are small only because `v_amp` and `v_trans`
are ~8 degrees apart, so neither alone captures it. **So `+1.05` is the scaling/time-shift
mode, not a physical instability, and the constraints correctly remove it** -- which is
also why it disappears from `L`.

(An earlier version of this measurement took `np.abs()` of a complex eigenvector before
projecting and reported a meaningless 100.0% for everything. Complex eigenvectors need
`vdot`, not `abs`.)

## What is open: the constrained spectrum is NOT converged in N

| N | leading eigenvalue of L |
|---|---|
| 28 | +6.92870 |
| 36 | +0.42553 -0.60488i |
| 44 | +0.75463 |

Wildly inconsistent -- these resolutions do not resolve it. The grid-scale modes
(`|Im| ~ N`, count growing with N) are also still present and must be separated as before.

**So the physical question -- is this profile stable modulo its symmetries? -- remains
open, and is now limited purely by resolution**, not by formulation. That is a much better
place than section 15 or 18, where the formulation itself was wrong.

## Standing results (unaffected by the above)

- **Converged self-similar profile**: `||F|| = 2.1e-12`, alpha self-consistent to 7.9e-7,
  `alpha = -0.342108` against Chen-Hou's `-0.342400` (**0.085%**).
- The corner frame, the constraint formulation, and the Newton solve are all validated by
  that convergence and by the symmetry mode landing at `+1.05 ~ 1`.

NEXT: push N (56, 64, 80) on the CONSTRAINED operator with the grid-scale modes filtered
by their `|Im| ~ N` scaling, and watch whether a low-`|Im|` eigenvalue converges. The
Newton solve is cheap; the dense Jacobian is the cost, so an Arnoldi on `L` with a shift
may be needed above N ~ 56.

# 23. RESOLUTION STUDY — what holds up and what does not

## The admissibility escape hatch is CLOSED

Chen-Hou's `eq:normal_vanish` restricts perturbations to `omega = O(|x|^2)`,
`theta_x, theta_y = O(|x|^2)` -- one order FASTER than the profile's own vanishing. In the
corner frame that is `dOt_xi(0,b) = 0` and `dBt_xixi(0,b) = 0` for ALL beta (2*nb linear
conditions, versus the two POINT conditions the gauge imposes). Galerkin-restricting the
operator to that subspace:

| N | dim -> restricted | invariance residual | UNRESTRICTED lead | RESTRICTED lead |
|---|---|---|---|---|
| 28 | 1458 -> 1404 | 0.005 | +6.9075 | +6.8677 |
| 36 | 2450 -> 2380 | 0.017 | +0.4250 +- 0.6044i | +0.4258 +- 0.6038i |
| 44 | 3698 -> 3612 | 0.007 | +0.7585 | +0.7567 |

The small invariance residual means the restriction is a VALID Galerkin projection (the
admissible space is nearly invariant, as the parity/order structure predicts). And
removing 54-86 dimensions changes the leading eigenvalue in the third digit. **Restricting
to their space does NOT remove the unstable modes**, so "our spectrum lives on a larger
space" does not explain the disagreement.

The earlier single-mode admissibility check was also inconclusive on its own: it flagged
EVERY mode inadmissible (corner powers 0.97-1.43), which means the discretization was
producing no admissible modes at all -- exactly why the subspace had to be constructed
explicitly rather than tested for after the fact.

## The profile itself is NOT converged in N — retract the 0.085%

| N | first xi node | 1/g there | c_l | alpha | vs reference |
|---|---|---|---|---|---|
| 28 | 0.08452 | 12.3 | 3.280970 | -0.341876 | 0.15% |
| 36 | 0.05032 | 20.4 | 3.063185 | -0.342108 | 0.085% |
| 44 | 0.03335 | 30.5 | 3.118043 | **-0.337819** | **1.3%** |

`alpha` gets WORSE from N=36 to N=44, and `c_l` moves 1.9% -> 3.7% off. **The
"alpha = -0.342108, 0.085%" reported in section 20 was a single-N result and does not
survive refinement.** The Newton residual is `~1e-12` at every N, so this is not a
convergence failure of Newton -- the DISCRETE FIXED POINT itself moves with N.

## The `1/g` hypothesis is REFUTED, and the answer is the opposite end

`1/g` at the first node grows like `N^2` (12.3 -> 20.4 -> 30.5), so corner amplification
was the natural suspect. Interpolating the N=44 profile onto the N=36 grid says otherwise:

| xi band | r | max rel diff |
|---|---|---|
| [0, 0.2) | 0 - 0.22 | **1.1e-4** |
| [0.2, 1) | 0.22 - 1.7 | 9.3e-3 |
| [1, 3) | 1.7 - 19 | 1.2e-2 |
| [3, 8) | 19 - 3e3 | 2.6e-2 |
| [8, 15) | 3e3 - 3e6 | 4.1e-2 |
| [15, 25) | 3e6 - 7e10 | **4.3e-2** |

The difference is SMALLEST at the corner and LARGEST in the far field -- a factor of 400
the other way. **The corner machinery is the best-behaved part of the whole
construction**; the resolution dependence lives at the outer end.

That points the next investigation at the far field: `XMAX = 25`, the outflow treatment
(nothing imposed on `Ot, Bt` there, which is correct for `c_l > 0`), and the
`d_xi Pt = 0` condition -- plus the fact that with `alpha0` fixed inside the substitution,
any residual `(c_w - c_l alpha0)` acts as a forcing proportional to `Ot` that the far
field feels directly. The outer alpha loop drives that to `~1e-7` at N=36; whether it does
so at every N has not been checked and is the first thing to verify.

## Standing, unaffected

- The corner-inclusive frame and its gates (section 17).
- Newton converges to `||F|| ~ 1e-12` at every resolution tried.
- The `+1.05` mode of the unconstrained operator is a SYMMETRY direction: 0.993 / 0.999 /
  1.000 overlap with `span{v_amp, v_trans}` against a ~0.03 random control.
- The constrained-operator construction `L = [I - B(Cg B)^-1 Cg] A` with exact `B` and
  `Cg`.

## NOT established, and not to be claimed

The stability of this profile. The constrained spectrum's leading eigenvalue is
`+6.87 / +0.43 +- 0.60i / +0.76` at N = 28 / 36 / 44 -- not converged, on a profile that
is itself not converged. No statement about Chen-Hou's or Liu's stability results is
supported until both converge.

# 24. THE ALPHA LOOP IS FINE — NEWTON BREAKS AT N=56

## The alpha loop converged at every N (hypothesis refuted)

| N | iters | final gap | alpha |
|---|---|---|---|
| 28 | 4 | 1.55e-09 | -0.34187584 |
| 36 | 5 | 2.50e-09 | -0.34210807 |
| 44 | 4 | 1.70e-09 | -0.33781927 |

So section 23's suspicion -- that N=44's alpha was a loop that had not closed -- is WRONG.
The loop closes to `~1e-9` everywhere. But the FIRST-iteration gap, starting from the
reference alpha, is telling: **5.2e-4 / 2.8e-4 / 4.6e-3**. N=44 jumps 16x further, so its
fixed point is genuinely displaced, not under-iterated.

## Newton FAILS at N=56 — and that voids the N=56 spectrum

| N | dim | \|\|F\|\| | c_l | alpha | wall |
|---|---|---|---|---|---|
| 36 | 2450 | 2.09e-12 | 3.063185 | -0.342108 | 1.1 min |
| 44 | 3698 | 2.63e-12 | 3.118043 | -0.337819 | 4.8 min |
| **56** | **6050** | **1.80e-02** | 3.086393 | -0.341039 | 27.4 min |

`1.8e-2` is ten orders off the other two. **The N=56 profile is not a solution, so its
spectrum is meaningless** -- and the "which eigenvalues recur" table was comparing two
converged profiles against a non-converged one, which is why nothing matched (distances
0.06-0.83 across the board). Discard the N=56 row entirely.

## Why Newton is likely breaking, and the fix

The Jacobian is FINITE-DIFFERENCED with `eps = 1e-6 * scale`. The operator contains
Chebyshev boundary rows scaling like `N^2` (`Dx`) and `N^4` (`Dx2` inside the Poisson
operator), plus `1/g` at the first node growing like `N^2` (12.3 / 20.4 / 30.5 / 51 at
N = 28 / 36 / 44 / 56). FD accuracy degrades as those grow, and past some N the Jacobian is
too inaccurate for Newton to take a good step.

**The RHS is a POLYNOMIAL (quadratic) in the fields**, so the Jacobian is available
ANALYTICALLY and no finite differences are needed at all:

  - `Pt = Poisson(Ot)` is LINEAR in `Ot`, with `dPt/dOt = -L_poisson^-1 (g^2 .)` -- and
    `L_poisson` is already prefactored, so this costs one existing back-substitution.
  - `advO, advB` are BILINEAR in `(Pt, Ot)` and `(Pt, Bt)`.
  - `srcO` is LINEAR in `Bt`.
  - `B` and `Cg` are already exact (section 22).

That removes the last FD block, kills the `eps` sensitivity, and makes each Newton step
cheaper than the current `2n` residual evaluations. It is the single change most likely to
unlock N >= 56.

Cheaper thing to try first: CONTINUATION -- solve at N=44, interpolate onto the N=56 grid,
and use that as the Newton seed instead of the reference interpolation. If N=56 then
converges to `~1e-12`, the problem was the seed's basin and not the Jacobian.

## Status of the physics: unchanged and open

Converged profiles exist only at N <= 44, and across those three `alpha` reads
`-0.34188 / -0.34211 / -0.33782` -- not converging. The constrained spectrum's leading
eigenvalue reads `+6.87 / +0.43 +- 0.60i / +0.76`. **Nothing about stability is
established**, and the blocker is now specifically: Newton must work at higher resolution.

# 25. THE ANALYTIC JACOBIAN — and the FD blocker confirmed

## The RHS is quadratic, so the Jacobian is closed-form

With `dP = Poisson(dOt)` (LINEAR in `dOt`, one back-substitution on the already
prefactored operator):

    d(advO) = (dP_x + mu dP) Ot_b + (Pt_x + mu Pt) dOt_b
              - dP_b (Ot_x + a0 Ot) - Pt_b (dOt_x + a0 dOt)
    d(srcO) = G cos b (dBt_x + (1+2a0) dBt) - sin b dBt_b
    d(advB) = (dP_x + mu dP) Bt_b + (Pt_x + mu Pt) dBt_b
              - dP_b (Bt_x + (1+2a0) Bt) - Pt_b (dBt_x + (1+2a0) dBt)

**Both** terms of each bilinear product appear; keeping only the `dOt`/`dBt` ones and
dropping the `dP` ones is the classic frozen-velocity Jacobian error.

### Gated against finite differences where FD is still trustworthy

| N | dim | rel \|\|A_exact - A_fd\|\| | max rel | leading eigenvalue | speed |
|---|---|---|---|---|---|
| 20 | 722 | 1.60e-08 | 7.2e-09 | identical to 6 dp | 2.08x |
| 28 | 1458 | 1.92e-09 | 1.8e-10 | identical to 6 dp | 2.04x |

Exact AND cheaper -- one Poisson solve per column instead of two residual evaluations.

## Newton with the FULL exact Jacobian

    J = [[ A , B ],
         [ Cg, 0 ]]

all three blocks closed-form (`A_exact`, `exact_B`, `exact_Cg`; lower-right is 0 because
the constraints do not depend on `c`). No finite differences anywhere.

| N | \|\|F\|\| | c_l | alpha | wall |
|---|---|---|---|---|
| 28 | 1.05e-12 | 3.280970 | -0.34187584 | 0.3 min |
| 36 | 2.25e-13 | 3.063185 | -0.34210807 | 1.1 min |

Same fixed points as FD Newton found (`c_l` and `alpha` agree to all printed digits), at
roughly a third of the wall time.

## CONTINUATION IS RULED OUT — it really is the Jacobian

Seeding N=56 from the CONVERGED N=44 profile (cubic interpolation, alpha carried over)
and running FD Newton: the residual stalls at **3.78e-2**, damping collapses to 0.031, and
the linesearch fails outright. Continuation gives a better starting residual (6.8e-2
against the reference seed's path) and still cannot descend.

So the N>=56 failure is NOT a basin problem and NOT a seed problem. **It is the
finite-difference Jacobian**, exactly as predicted from the `N^2` (`Dx`), `N^4` (`Dx2`) and
`1/g ~ N^2` scalings that make `eps = 1e-6 * scale` stop resolving it.

That is a clean, falsifiable prediction made in section 24 and confirmed by two
independent routes (continuation failing, exact Jacobian succeeding at low N). The exact
Jacobian at N >= 44 is running.

# 26. TWO WRONG PREDICTIONS — the N>=56 failure is neither the Jacobian nor the Poisson solve

## Prediction 1 (section 24-25): the finite-difference Jacobian. WRONG.

The analytic Jacobian was built and gated properly (rel `1.6e-08` / `1.9e-09` against FD
at N=20/28, leading eigenvalues identical to 6 dp, 2x faster), Newton was rebuilt on the
fully exact `J = [[A,B],[Cg,0]]`, and it reproduces the FD fixed points at N=28/36/44 to
every printed digit at a third of the wall time:

| N | \|\|F\|\| | c_l | alpha | wall |
|---|---|---|---|---|
| 28 | 1.05e-12 | 3.280970 | -0.34187584 | 0.3 min |
| 36 | 2.25e-13 | 3.063185 | -0.34210807 | 1.1 min |
| 44 | 6.94e-13 | 3.118043 | -0.33781927 | 3.9 min |
| **56** | **1.73e-02** | 3.105823 | -0.34258999 | 30.9 min |

**N=56 fails with the EXACT Jacobian too**, at essentially the same residual FD gave
(1.73e-2 against 1.8e-2). The exact Jacobian is a genuine improvement -- it is exact, it
is faster, and it should be kept -- but it does NOT unlock N >= 56.

## Prediction 2: the Poisson operator's conditioning. ALSO WRONG.

Its `d_xixi` coefficient is `g^2 ~ xi^2 ~ N^-4` at the first node while `Db2` is `O(N^2)`,
so anisotropy was the natural suspect. Manufactured-RHS residual of the prefactored LU:

| N | 28 | 36 | 44 | 48 | 52 | 56 | 64 |
|---|---|---|---|---|---|---|---|
| rel residual | 4.2e-14 | 2.8e-14 | 6.0e-14 | 4.3e-14 | 8.3e-14 | 5.1e-14 | 7.3e-14 |

FLAT. The Poisson solve is clean to `1e-14` at every resolution tried.

## Also ruled out: continuation

Seeding N=56 from the CONVERGED N=44 profile stalls at `3.78e-2` with the linesearch
failing and damping collapsing to 0.031 -- starting from a BETTER residual than the
reference seed and still unable to descend. Not a basin problem, not a seed problem.

## What is left

Ruled out so far: the FD Jacobian, the Poisson solve, the seed/basin, the alpha loop
(section 24: converges to ~1e-9 at every N). Still open, and now being bracketed at
N = 44 / 48 / 52 / 56 with `cond(J)` measured at the first Newton step:

  - conditioning of the FULL Newton system (not the Poisson block) -- `1/g` at the first
    node reaches 49.6 at N=56 and multiplies the whole advection/source term, so the
    Jacobian may simply become too ill-conditioned for a direct solve even when every
    block is exact;
  - or the discrete problem genuinely having no nearby solution at that resolution.

**METHOD NOTE.** Two confident predictions in a row, both wrong, both eliminated in one
measurement each. The pattern worth keeping is that each was stated sharply enough to be
killed cheaply -- the FD hypothesis by building the exact Jacobian, the Poisson hypothesis
by one manufactured-RHS solve. What should stop is leading with the prediction; the
bracket-and-measure should come first.

# 27. RESOLUTION STRUCTURE — N is independent, and what that buys

## Each N is its own problem

`F_N(x) = 0` is a separate nonlinear system per resolution with its own solution `x_N`.
There is NO requirement to walk `28 -> 36 -> 44 -> 56` in order, and nothing about a
coarse solve is needed to reach a fine one. The ONLY coupling is the SEED:

  - default: interpolate Chen-Hou's reference profile onto the N grid
  - continuation: interpolate a CONVERGED solution from a neighbouring N

Continuation 44 -> 56 was tested and FAILED (stalls at 3.78e-2 from a better starting
residual than the reference seed), so the seed is not the constraint here. That also means
resolutions can be attempted in any order, or skipped.

## Cost

Unknowns `n = 2 N^2` (minus boundary rows). Per Newton step: a Jacobian build costing
`n` Poisson solves, then a DENSE solve at `O(n^3) = O(N^6)`. The spectrum is another
`O(N^6)` eigendecomposition. Measured wall time (exact Jacobian, full alpha loop):

| N | dim | wall |
|---|---|---|
| 28 | 1458 | 0.3 min |
| 36 | 2450 | 1.1 min |
| 44 | 3698 | 3.9 min |
| 56 | 6050 | 30.9 min |

Extrapolating at `N^6`: N=64 ~ 68 min, N=80 ~ 4.4 h. That is the practical ceiling for
dense linear algebra on this machine, and it is why making the Jacobian build 2x faster
matters less than it sounds -- the dense solve dominates.

## The experiment this structure enables

Because N is independent, **56 can simply be SKIPPED**, and doing so distinguishes two
very different diagnoses:

  - N=60/64 CONVERGE  -> N=56 is an ISOLATED bad case (a resonance at that particular
                         grid), the method is sound, and the resolution study can proceed
                         by avoiding it
  - N=60/64 ALSO FAIL -> a genuine THRESHOLD above N ~ 44, i.e. something degrades
                         systematically, and no amount of seeding or Jacobian accuracy
                         will help

Running now. If it is a threshold, the dense-direct approach is finished around N=44 and
the route forward is iterative (matrix-free Newton-Krylov with a preconditioner, and
shift-invert Arnoldi for the spectrum) rather than more of the same.

# 28. THE EIGENVALUES ARE NOT MEASURABLE — a pseudospectral verdict

## The measurement

Eigenvalue condition number `kappa(lam) = 1/|y^H x|` (left/right eigenvectors, unit
norm): a perturbation of size `eps` moves `lam` by up to `kappa * eps`. Since CHANGING N
IS A PERTURBATION of the operator, `kappa` says directly whether an N-study can ever
converge.

| N | \|\|F\|\| | normality dep. | MEDIAN kappa | MAX kappa |
|---|---|---|---|---|
| 28 | 1.0e-12 | 1.4141 | **2.01e+15** | 7.98e+17 |
| 36 | 2.3e-13 | 1.4131 | **4.62e+15** | 4.84e+18 |
| 44 | 6.9e-13 | 1.4140 | **3.63e+15** | 2.13e+18 |

**The MEDIAN eigenvalue has `kappa ~ 3e15`** -- a perturbation at MACHINE EPSILON
(2.2e-16) moves it by ~0.7. The typical eigenvalue of this operator carries no
information whatsoever.

## And the modes I chased for three turns are the worst of them

| N | mode | kappa | moved by 1e-10 |
|---|---|---|---|
| 36 | **+0.42555 +- 0.60475i** | **7.14e+14** | 7.1e+04 |
| 44 | **+0.11082 +- 0.76424i** | **3.66e+14** | 3.7e+04 |
| 28 | +6.90774 (real) | 3.27e+04 | 3.3e-06 |
| 44 | +0.75186 (real) | 4.64e+05 | 4.6e-05 |
| 36 | -0.07881 (real) | 7.60e+05 | 7.6e-05 |
| 44 | -0.35091 (real) | 4.29e+06 | 4.3e-04 |

A clean split: the **REAL** eigenvalues are moderately conditioned (`1e4`-`1e6`), the
**COMPLEX PAIRS are catastrophic** (`~1e14`-`1e15`). That is the classic signature of a
NEARLY DEFECTIVE operator -- two nearly-coalescing real eigenvalues forming a spurious
complex pair with an enormous condition number.

**So the "unstable oscillatory mode" tracked since section 21 (+0.43 +- 0.60i, period
~9.8, then +0.11 +- 0.76i) is a NUMERICAL ARTIFACT of a nearly-defective, strongly
non-normal operator. It is not physics.** Retract every reading of it.

## This also explains the resolution study that would not converge

Sections 23-27 chased `+6.87 / +0.43 +- 0.60i / +0.76` across N and treated the scatter as
"unresolved". With `kappa ~ 1e4` on the best-conditioned mode and a discretization
difference of `~1e-4` between grids, the expected movement is `~3` -- which IS the
observed scatter. **The N-study could never have converged.** More resolution perturbs
those eigenvalues as much as it resolves them. That is three turns of work explained, and
retired, by one cheap measurement.

Note the constraint projection is what made this so bad: departure from normality is
**1.414 for the CONSTRAINED operator `L`** against **0.10 for the unconstrained `A`**. The
gauge projection is necessary AND it is what pushes the operator to near-defectiveness.

## The right tools, and where they come from

For a non-normal operator the eigenvalues are the wrong object. The correct ones are
well-conditioned even when eigenvalues are not:

  - **pseudospectra** (Trefethen -- the same source as the Chebyshev differentiation
    matrices used here): the region where `||(sI - L)^-1||` is large, i.e. where
    eigenvalues of NEARBY operators live.
  - **resolvent analysis** (hydrodynamic stability / transition to turbulence): singular
    values of `(sI - L)^-1`. Developed precisely because pipe flow is linearly STABLE at
    every Reynolds number where it demonstrably becomes turbulent -- eigenvalues said one
    thing, the flow said another, and the resolution was that non-normal operators
    amplify enormously with every eigenvalue in the left half-plane.
  - **transient growth / optimal perturbations**: `max_t ||e^{Lt}||`, what actually grows
    over a finite horizon.
  - **randomized SVD / sketching** (theoretical CS): dominant subspaces from a few
    matrix-vector products, which also removes the `O(N^6)` dense wall.

The uncomfortable symmetry: this is the SAME failure mode that made pipe-flow stability
theory wrong for a century, appearing in a problem two steps away from it. Both directions
are now open -- unstable eigenvalues that do not matter, or stable ones hiding large
transient growth.

**NEXT: stop computing eigenvalues. Compute `||(sI - L)^-1||` on a grid in `s`, and
`||e^{Lt}||` against `t`.** Both are well-conditioned, both answer the physical question
directly, and neither needs the N-convergence that was never achievable.

# 29. THE OPERATOR ITSELF IS UNRESOLVED — the real blocker

## Even the well-conditioned quantities do not converge

`G(t) = ||e^{Lt}||` and `R(s) = 1/sigma_min(sI - L)` are SINGULAR-VALUE quantities, hence
Lipschitz in the matrix: a perturbation of size `eps` moves them by at most `eps`. They
CANNOT be hypersensitive the way eigenvalues are. So they were the honest test.

| quantity | N=28 | N=36 | relative difference |
|---|---|---|---|
| max transient growth | **3.2685e+40** | **1.3852e+07** | **2.4e+33** |
| resolvent norm, real axis | -- | -- | median **0.88**, max 7.65 |

Thirty-three orders of magnitude on a Lipschitz quantity means **the operators are not
close**. `L(N=28)` and `L(N=36)` are not two accuracies of the same operator; they are
DIFFERENT operators. At N=28 the growth is `~e^{6.9 t}` (tracking that resolution's
+6.908 eigenvalue, which is only moderately conditioned at kappa=3.3e4); at N=36 nothing
like it exists.

## What this actually means

The eigenvalue ill-conditioning of section 28 is REAL but SECONDARY. The primary fact is
that **the linearised operator is not resolved at N <= 52**, which is why nothing computed
from it converges -- eigenvalues, resolvent, or transient growth alike. It is consistent
with the PROFILE not converging either (`alpha` = -0.34188 / -0.34211 / -0.33782 at
N = 28/36/44).

Combined with section 26's ceiling -- dense direct Newton dies at `N ~ 52`,
`cond(J) ~ 1e11`, not fixable by a better Jacobian (exact) or a better seed (continuation
failed) -- the position is:

    RESOLUTION NEEDED FOR CONVERGENCE  >  RESOLUTION REACHABLE BY DENSE DIRECT METHODS

That is the blocker, stated plainly. Everything else in this document is built and gated:
the corner frame, the gauge, the constraints, Newton, the exact Jacobian, the admissible
subspace. None of it is wrong. It simply cannot be run at a resolution where the answer
stops moving.

## ARTIFACT to fix in `polar_resolvent.py`

`R(0) ~ 1.6e15` at every N is MY OWN construction, not the operator: the projection `L`
has exactly TWO zero eigenvalues by design (`P` has rank `n-2`), so `sigma_min(0*I - L)`
is zero to roundoff. Exclude `s = 0`, or measure the resolvent on the subspace orthogonal
to the projection's null space.

## Routes that could actually break this

1. **Iterative Newton-Krylov with a preconditioner.** Never forms or inverts the dense
   Jacobian, so the `N ~ 52` / `cond ~ 1e11` ceiling does not apply. The Poisson operator
   is already prefactored and is the natural preconditioner.
2. **A better discretization rather than more points.** The residual has concentrated in
   `xi in [2,15]` (r ~ 7 to 3e6) from the very beginning, and Chebyshev puts its FEWEST
   nodes there. A mapped or multi-panel grid tuned to that band could resolve the same
   physics at far lower N -- and KTE mapping was tried and did not help, so this needs the
   panel version, not another map.
3. **Randomized SVD / sketching** for `G(t)` and `R(s)`: dominant subspaces from
   matrix-vector products only, removing the `O(N^6)` wall for the diagnostics even if
   Newton still needs a direct solve.

Route 1 is the one that changes the ceiling. Routes 2 and 3 reduce what is needed and what
it costs.

## METHOD NOTE

The sequence "eigenvalues do not converge -> they are ill-conditioned -> use
well-conditioned quantities instead -> THOSE do not converge either" is the useful shape
here. Each step was a real measurement that killed a real hypothesis, and the last one
located the blocker one level deeper than the previous three. The eigenvalue-conditioning
result is still correct and still worth having; it was simply not the bottom of the stack.

# 30. THE FULL ALPHA TABLE, and a bad patch around N ~ 52-56

Every converged Newton solve, exact Jacobian, outer alpha loop closed to ~1e-9:

| N | \|\|F\|\| | alpha | vs reference -0.34240009 |
|---|---|---|---|
| 28 | 1.05e-12 | -0.341876 | +0.15% |
| 36 | 2.25e-13 | -0.342108 | +0.09% |
| 44 | 6.94e-13 | -0.337819 | -1.34% |
| **52** | **3.48e-12** | **-0.360552** | **+5.30%** |
| 56 | 1.73e-02 | -- | NEWTON FAILED |
| 60 | 1.83e-12 | -0.346251 | +1.13% |
| 64 | 1.89e-12 | -0.347176 | +1.40% |

**N = 52 and 56 are a bad patch**: 56 fails outright, and 52 converges but to the worst
alpha (+5.3%) with a residual an order worse than its neighbours (3.5e-12 against ~1e-12).
Excluding that pair, alpha spans -1.3% to +1.4% -- still not converging, and drifting
DOWNWARD at the two highest resolutions rather than settling on the reference.

Sixteen times more unknowns from N=28 to N=64 buys nothing. This is a discretization-
QUALITY problem, not a resolution problem, and it is the live blocker.

# 31. AN ARTIFACT I KEEP RE-DISCOVERING — the projection's null space

`cond(L) = 1.003e+16` at N=36 (`sigma_min = 2.03e-11`, `sigma_max = 2.03e+05`) looks like
a numerically singular operator and would explain everything at once. **It is mostly MY
OWN CONSTRUCTION.** `P = I - B(Cg B)^-1 Cg` has rank `n-2` BY DESIGN, so `L = PA` has TWO
EXACT ZERO singular values; the measured `2e-11` is those zeros in roundoff. Reporting
`sigma_max/sigma_min` therefore measures the GAUGE, not the physics.

This is the third time the same artifact has surfaced:
  - `R(0) ~ 1.6e15` in `polar_resolvent.py` (section 29)
  - `cond(L) ~ 1e16` here
  - and it inflates any "median kappa" computed over ALL modes, since the two null
    directions have undefined eigenvector pairing

**RULE: whenever a quantity is computed from `L`, exclude the projection's 2-dimensional
null space first** -- use `sigma[-3]` not `sigma[-1]`, restrict the resolvent to
`range(P)`, and drop the null modes before any spectral statistic. The genuine
conditioning of the operator on its own range is the number that matters, and it has not
yet been measured cleanly.

# 32. THE DRIFT IS STRUCTURED — and it names its own fix

## The measurement

Overlap of the unit fixed-point drift direction with a K=40 subspace of an n=2450 space
(chance level `sqrt(40/2450) = 0.128`):

| drift | \|dx\| | 40 smallest singular dirs | 40 highest-kappa dirs | random control |
|---|---|---|---|---|
| N=36 -> 44 | 9.03 | **0.9898** | 0.7871 | 0.1105 |
| N=44 -> 52 | 32.0 | **0.9966** | 0.7922 | 0.1174 |
| N=36 -> 52 | 23.4 | **0.9966** | 0.7836 | 0.1276 |

**The alpha wandering is NOT noise.** It lives almost entirely in the 40 smallest singular
directions of `L`, exactly as `dx ~ -J^-1 dF` predicts, at 8x the chance level. The
high-kappa subspace also beats chance strongly (0.78 vs 0.12), so the ill-conditioning and
the drift are the same phenomenon seen two ways.

Conditioning, with the projection's null space properly excluded:
`sigma_min = 1.443e-04`, `cond = 1.41e9` (NOT the 1.003e16 the raw ratio reports).
Smallest genuine singular values: 2.3e-3, 1.4e-3, 6.4e-4, 1.4e-4, then the two
construction zeros at 9.0e-11 and 2.0e-11.

## What those directions ARE

| property | first 12 near-null right singular vectors |
|---|---|
| **Ot-fraction** (vorticity vs B) | **0.68 - 0.99** |
| **beta-peak / (pi/2)** | **0.950 - 0.997** |

**Vorticity perturbations concentrated at the SYMMETRY AXIS.** One field, one location --
not generic numerical fuzz. (Their `xi` distribution is broad, with weight at both
`xi < 2` and `xi > 20`, so this is an ANGULAR problem, not a radial one.)

## Why the axis is where the operator loses grip

At `beta = pi/2`: `Psi = 0`, so `Pt = 0` and the beta-advection coefficient
`(Pt_xi + mu Pt)` VANISHES; and `Om` itself vanishes linearly. The operator therefore has
almost no control over vorticity there -- small singular values -- and every bit of
discretization error is funnelled into precisely those directions, moving the fixed point
and dragging `alpha` with it.

Implicated as well: the beta grid stops at `pi/2 - 1e-3`, an arbitrary offset introduced
early for interpolation safety. The last node is pinned; the nodes just inside it are
weakly constrained. That is the soft spot.

## THE FIX — the same move that fixed the corner, applied to beta

`r = 0` was pathological until the vanishing was absorbed into the coordinate
(`xi = ln(1+r)`) and the substituted fields became O(1). The axis needs the identical
treatment. Measured (section 8): `Om ~ (pi/2 - beta)^1`, `B ~ (pi/2 - beta)^2`. So set

    Ot = cos(beta) * Ot_hat ,      Bt = cos^2(beta) * Bt_hat

The new unknowns are O(1) at the axis, the operator has grip on them, and the angular
regularity is absorbed into the variables instead of being fought by the solver -- exactly
as `Pt = 0` at the corner stopped being imposed and started EMERGING once the frame was
right. It should also let the beta endpoint go to `pi/2` exactly and retire the `1e-3`
offset.

**This is the first time in the whole profile route that a measurement has named its own
fix.** Everything before it was elimination.

## METHOD NOTE — where this came from

The user asked whether two noisy quantities (fixed-point drift, and the kappa
"directional flow") might correlate, with the caveat "as long as it doesn't degrade it".
The answer required a RANDOM-SUBSPACE CONTROL to mean anything -- any two noisy vectors
show some overlap, and without the 0.128 baseline the 0.99 would have been unfalsifiable.
With the control it is an 8-sigma-scale result and it localised a three-turn blocker to
one field at one boundary.

# 33. THE AXIS SUBSTITUTION FAILED — and the failure re-diagnoses the problem

## What was tried

`Ot = cos(b) Ot_hat`, `Bt = cos^2(b) Bt_hat`, applied at the pack/unpack boundary so it
is a pure change of unknowns, i.e. the diagonal preconditioner `D^-1 J D`. Motivated by
section 32: the drift lives in near-null directions that are VORTICITY AT THE AXIS, and
the measured orders there are `Ot ~ eps^1.10`, `Bt ~ eps^2.25`. `Pt ~ eps^1.03` was
verified FIRST so the `tan(b)` term the division creates stays finite.

| N | axis sub | \|\|F\|\| | alpha | cond(J0) |
|---|---|---|---|---|
| 36 | off | 2.51e-12 | -0.34212374 | 3.19e+09 |
| 36 | **ON** | **7.47e-03** | -0.34235302 | 2.69e+08 |
| 44 | off | 1.68e-12 | -0.33785061 | 1.09e+10 |
| 44 | **ON** | **7.25e-03** | -0.34230697 | 9.75e+09 |
| 52 | off | 1.35e-03 | -0.36044243 | 6.26e+09 |
| 52 | **ON** | **7.33e-03** | -0.34240009 | 2.61e+11 |

**It broke Newton.** And the `alpha` column with the substitution ON is FAKE: those values
sit at (or beside) `-0.34240009`, which is the reference `c_w/c_l` used to INITIALISE the
solve. Newton never moved them. At N=52 it reproduces the reference to all 8 digits, which
is the giveaway. **Never read a parameter off a run whose residual did not converge** --
it is the initial condition wearing the answer's clothes.

## Why it failed, and what that reveals

`Ot` is NOT small near the axis -- it PEAKS there. From `beta = 0` outward it rises from
1.125 to **4.89 at `eps = 0.03`**, then falls to zero across the final 0.03 radians.
Dividing by `cos b` therefore amplifies exactly where the field is LARGEST: `Ot_hat` spans
1.1 to ~228 (a factor 200) against `Ot`'s original 1.1 to 4.9 (a factor 4.5). The
"preconditioner" made the dynamic range **45x worse**.

**So the near-null directions are not weakly SCALED, they are under-RESOLVED.** The
near-axis structure is a boundary layer of width ~0.03 rad. At `Nb = 36` the Chebyshev
spacing near `beta = pi/2` is `~(pi/2)(pi/35)^2/2 ~ 0.006`, i.e. only about **5 nodes
across the layer**. That is a resolution problem in BETA, and `Nb` has been tied to `Nx`
for the entire project -- every resolution study so far varied them together, so the beta
direction was never tested independently.

## The corrected experiment

Fix `Nx`, sweep `Nb` alone. If the axis layer is the blocker, `alpha` moves with `Nb` and
settles at fixed `Nx`. If it does not, the layer is not the cause and the near-null
directions have some other origin.

This is the third distinct explanation for the same non-convergence (FD Jacobian ->
conditioning -> under-resolved axis layer). The first two were killed by measurement; this
one is being tested the same way.

# 34. BETA IS CONVERGED — the problem is RADIAL

## The sweep that settles it

`Nx` and `Nb` had been tied together for the entire project, so the beta direction had
never been tested on its own. Fixing `Nx` and tripling `Nb`:

| Nx | Nb | dim | \|\|F\|\| | alpha | vs ref |
|---|---|---|---|---|---|
| 36 | 36 | 2450 | 2.25e-13 | -0.34210807 | 0.085% |
| 36 | 52 | 3570 | 2.10e-13 | -0.34210440 | 0.086% |
| 36 | 72 | 4970 | 2.66e-13 | -0.34210093 | 0.087% |
| 36 | 96 | 6650 | 3.65e-13 | -0.34209736 | 0.088% |
| 44 | 72 | 6106 | 2.85e-13 | -0.33781876 | 1.338% |
| 44 | 96 | 8170 | 3.15e-13 | -0.33781855 | 1.338% |

**alpha is CONVERGED in `Nb` to 1e-5** (2e-7 at Nx=44). Change `Nx` from 36 to 44 and it
jumps 1.25%. **The non-convergence is entirely RADIAL.** The under-resolved-axis-layer
hypothesis of section 33 is dead.

## A correction to section 32's framing

Section 32 said the drift measurement "named its own fix". **It did not** -- it named
where the RESPONSE lives, not where the error comes from. `dx ~ -J^-1 dF` puts the
response along the operator's weakest directions REGARDLESS of the source of `dF`. So:

  - the near-null directions ARE vorticity at the axis (measured, 0.68-0.99 Ot-fraction,
    beta-peak 0.95-0.997) -- that is a true statement about `J`;
  - the drift DOES overlap them at 0.99 against a 0.128 control -- also true;
  - but the error DRIVING them is radial, as the `Nb` sweep now proves.

**Drift direction diagnoses the OPERATOR's weak directions; it does not localise the
error's SOURCE.** Those are different questions and I conflated them. The 0.99 overlap
remains a real and useful result about `J`; it was simply the wrong inference to hang a
fix on.

This also reconciles with the earliest localisation, which said the same thing and was
never contradicted: the N=36-vs-44 profiles differ by **4.3e-2 in the far field** and
**1.1e-4 at the corner**.

## Next

`Nb` is now known converged, so `XMAX` can be varied cleanly at fixed `Nx`:
  - alpha moves with `XMAX`  -> the OUTER BOUNDARY TREATMENT is the cause
  - alpha moves only with `Nx` at fixed `XMAX` -> radial RESOLUTION, and the fix is
    radial panels concentrated where the residual has always lived (`xi in [2,15]`),
    not a uniform increase and not another global map (KTE was tried, section 27).
