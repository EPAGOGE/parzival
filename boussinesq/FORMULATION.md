# FORMULATION -- 2D incompressible Boussinesq, Luo-Hou corner-blowup scenario

Status: formulation, pre-implementation. Written 2026-07-22.
Every load-bearing spectral claim below (transform conventions, derivative maps,
alias identities, quadrature exactness, parity closure, semi-discrete budgets,
Taylor-Green steadiness, gravity-wave dispersion) was verified numerically on a
throwaway N=64 harness before this document was committed; measured residuals
are quoted inline as `[measured: ...]`. Anything not so verified is labeled
ESTIMATE or UNVERIFIED.

## 0. Constitution (inherited from swarm_m1.py / audit.py discipline)

1. Gates before science. No scenario run counts until G1-G5 pass.
2. fp64 numpy/scipy is the primary engine. Any accelerated mirror (torch/MPS)
   is secondary and must pass a cross-backend gate against fp64, as in
   swarm_m1 gate 2.
3. No dense NxN^2 operator matrices. Everything spectral via scipy.fft
   (DST/DCT), O(N^2 log N) per RHS evaluation.
4. No wall-clocks and no randomness inside physics code paths. Adaptive dt is
   a deterministic function of the state. Output cadence is by step count.
5. Monitors are never steering targets (probe-is-not-the-loss law).
6. Meter-era honesty: tolerances below are either derived, or measured-then-
   tripwired; the first full-engine run recalibrates any ESTIMATE.

## 1. Equations and sign conventions

### 1.1 Pinned conventions

Domain coordinates x = (x1, x2), gravity acts along -x2 (down). Buoyancy
variable theta = g*alpha*(T - T0) has dimensions of acceleration (L/T^2);
warm fluid (theta > 0) is light. `Lap` below is the STANDARD signed Laplacian
Lap = d^2/dx1^2 + d^2/dx2^2 (negative-definite on our bases).

Velocity from streamfunction (2D perp-gradient):

    u = (u1, u2) = (-psi_x2, +psi_x1)                                   (1)

Scalar vorticity (curl in the +x3 sense):

    w = u2_x1 - u1_x2 = psi_x1x1 + psi_x2x2 = Lap(psi)                  (2)

Evolution equations:

    w_t  + u.grad(w)     = theta_x1 + nu  * Lap(w)                      (3)
    theta_t + u.grad(theta) =          kappa * Lap(theta)               (4)
    Lap(psi) = w,   u = (-psi_x2, psi_x1)                               (5)

Default nu = kappa = 0 (inviscid: the open-problem / proven-theorem regime).
nu, kappa > 0 is the viscous option; with the parity bases of Section 2 it
realizes stress-free (free-slip) walls.

### 1.2 Buoyancy-torque sign verification

Momentum form: u_t + u.grad(u) = -grad(p) + theta*e2 + nu*Lap(u). With gravity
-g*e2 and Boussinesq density rho = rho0*(1 - alpha*(T-T0)), subtracting the
hydrostatic base pressure leaves the buoyancy force +theta*e2: warm fluid is
pushed UP, against gravity. Correct.

Curl of the force: curl(theta*e2).e3 = d/dx1(theta) - d/dx2(0) = +theta_x1.
Hence the +theta_x1 in (3).

Physical check: a warm blob (theta peaked at a point) has theta_x1 < 0 on its
right flank -> generates w < 0 (clockwise) there; theta_x1 > 0 on the left
flank -> w > 0 (counterclockwise). Both cells lift fluid through the blob
center: hot fluid rises. Sign confirmed.

Energy check (Section 4): this sign convention yields
dE/dt = int(theta*u2) - nu*int(w^2), i.e. buoyancy does positive work on
rising warm fluid and viscosity strictly dissipates. Self-consistent.

### 1.3 Two sign issues in the task statement, caught and fixed

(a) Viscosity. The tasked form `w_t + u.grad(w) = theta_x1 - nu*Lap(w)`
together with `Lap(psi) = w` is internally inconsistent if `Lap` denotes one
operator: with the standard Laplacian (the one satisfying Lap(psi)=w in (2)),
`-nu*Lap(w)` is ANTI-dissipative (energy growth). The task's own budget
identity dE/dt = ... - nu*int(w^2) requires dissipation, i.e. +nu*Lap(w) with
the standard Laplacian (equivalently -nu*(-Lap) with the positive operator).
We pin: standard Laplacian everywhere, dissipation terms +nu*Lap(w),
+kappa*Lap(theta), as written in (3)-(4).

(b) Stratification sign in gate G3. The tasked rest state `theta = -B*x2`
labeled "stable" is UNSTABLE under the pinned convention (buoyancy +theta*e2):
a parcel displaced upward from theta0 = -B*x2 (B>0) is warmer than its new
surroundings and keeps rising (Rayleigh-Taylor/convective). The stable state
is theta0 = +B*x2 (theta increasing upward: light over heavy), and it is this
state that yields exactly the tasked dispersion sigma^2 = B*k1^2/|k|^2
(re-derived in G3). The sign slip is in the base state, not in the frequency
formula. (Under the opposite force convention -theta*e2, -B*x2 would be the
stable one; we do not use that convention.)

## 2. Geometry: box [0,pi]x[0,pi] via parity bases

### 2.1 Scenario mapping

Luo-Hou 3D axisymmetric Euler: cylinder wall at r=1, symmetry plane z=0,
singularity at the boundary ring (r,z)=(1,0). Standard 2D Boussinesq analog:
x1 <-> z (symmetry direction; the driving derivative theta_x1 corresponds to
the d/dz of the swirl term), x2 <-> 1-r (wall-normal). Our box:

- x2 = 0 : solid no-penetration wall           (u2 = 0 there)
- x1 = 0 : symmetry axis of the scenario        (u1 = 0 there; mirror flow)
- corner (0,0): the two lines meet; the flow develops a hyperbolic stagnation
  point there and the singular action is expected corner-adjacent.
- x1 = pi, x2 = pi: images of the same conditions under the parity extension
  (a second mirror and a second free-slip wall). The box is closed: u.n = 0 on
  all four sides.

### 2.2 Parity table

Realize the box by extending every field to the 2pi-periodic torus
[-pi,pi)^2 with a definite parity per direction; sine series = odd extension,
cosine series = even extension. Grid is cell-centered (Section 2.5), so no
grid point lies on a wall.

| field | parity in x1 | parity in x2 | series      | vanishes on          |
|-------|--------------|--------------|-------------|----------------------|
| psi   | odd  (sin)   | odd  (sin)   | sin (x) sin | all four sides       |
| w     | odd  (sin)   | odd  (sin)   | sin (x) sin | all four sides       |
| theta | even (cos)   | odd  (sin)   | cos (x) sin | x2 = 0, pi           |
| u1    | odd  (sin)   | even (cos)   | sin (x) cos | x1 = 0, pi           |
| u2    | even (cos)   | odd  (sin)   | cos (x) sin | x2 = 0, pi           |

Boundary verification: u2 in *(x)sin vanishes at x2=0,pi (no penetration at
both horizontal walls); u1 in sin(x)* vanishes at x1=0,pi (symmetry axes).
u.n = 0 on all of d(box). At the corner (0,0) both components vanish:
stagnation point. Near the corner psi ~ C*x1*x2 (leading sine-sine behavior),
so u ~ C*(-x1, +x2): hyperbolic.

### 2.3 The table is forced (uniqueness)

- No-penetration at x2=0 for all x1 <=> psi = const on the wall; in a parity
  basis that is psi odd in x2. (Choosing psi even in x2 gives u2 = psi_x1 in
  cos(x)cos, nonzero at the wall: rejected.)
- Symmetry axis at x1=0 <=> u1 odd in x1 <=> psi odd in x1.
- w = Lap(psi) has psi's parity: sin(x)sin. So w = 0 on all four sides.
- The vorticity equation then forces theta's class: theta_x1 must lie in
  w's class sin(x)sin, so theta is cos in x1 and sin in x2. No other
  assignment closes. In particular theta = 0 on both horizontal walls is
  FORCED, not chosen. Consequences flagged in Section 8.

### 2.4 Parity closure, term by term

Per-axis parity algebra: odd*odd = even, odd*even = odd, even*even = even;
d/dxi flips the parity in xi only; Lap preserves both parities.

Derived intermediate classes:
w_x1: cos(x)sin, w_x2: sin(x)cos, theta_x1: sin(x)sin, theta_x2: cos(x)cos.

Vorticity equation -- every term must be sin(x)sin (odd,odd):

| term        | x1 parity          | x2 parity          | class      | ok |
|-------------|--------------------|--------------------|------------|----|
| w_t         | odd                | odd                | sin(x)sin  | ok |
| u1*w_x1     | odd*even = odd     | even*odd = odd     | sin(x)sin  | ok |
| u2*w_x2     | even*odd = odd     | odd*even = odd     | sin(x)sin  | ok |
| theta_x1    | odd                | odd                | sin(x)sin  | ok |
| nu*Lap(w)   | odd                | odd                | sin(x)sin  | ok |

Temperature equation -- every term must be cos(x)sin (even,odd):

| term            | x1 parity          | x2 parity          | class      | ok |
|-----------------|--------------------|--------------------|------------|----|
| theta_t         | even               | odd                | cos(x)sin  | ok |
| u1*theta_x1     | odd*odd = even     | even*odd = odd     | cos(x)sin  | ok |
| u2*theta_x2     | even*even = even   | odd*even = odd     | cos(x)sin  | ok |
| kappa*Lap(theta)| even               | odd                | cos(x)sin  | ok |
| -B*u2 (G3 only) | even               | odd                | cos(x)sin  | ok |

Both PDEs map the parity classes into themselves: the classes are invariant
under the flow, and evolving only the reduced coefficient arrays is exact
(the box solution IS the restriction of the mirror-extended periodic
solution). [measured: full discrete RHS built this way conserves the class
structurally; equivariance gate G5 quantifies bug-freedom.]

### 2.5 Grid, transforms, wavenumber grids, inverse pairs

Cell-centered grid, identical in both directions:

    h = pi/N,  x_j = (j + 1/2)*h,  j = 0..N-1        (array axis 0 = x1)

scipy.fft type-II transforms on this grid are exactly the parity series:

- DST-II: basis sin(m*x_j), frequencies m = k+1 for slot k, i.e. m = 1..N.
  Wavenumber grid: `ks = np.arange(1, N+1)` (slot k holds frequency k+1).
- DCT-II: basis cos(m*x_j), frequencies m = k = 0..N-1.
  Wavenumber grid: `kc = np.arange(0, N)`.

Forward/inverse calls (all with `norm='ortho'`; under ortho the type-III
transform is the exact inverse of type-II, and scipy's idst/idct(type=2) are
precisely those inverses):

    # sine axis                                  # cosine axis
    F = sfft.dst (f, type=2, axis=a, norm='ortho')   F = sfft.dct (f, type=2, axis=a, norm='ortho')
    f = sfft.idst(F, type=2, axis=a, norm='ortho')   f = sfft.idct(F, type=2, axis=a, norm='ortho')

Per-field 2D transforms (axis 0 = x1, axis 1 = x2):

    w, psi   : sfft.dstn(f, type=2, norm='ortho')            # sin (x) sin
    theta,u2 : dct(axis=0) then dst(axis=1), type=2, ortho   # cos (x) sin
    u1       : dst(axis=0) then dct(axis=1), type=2, ortho   # sin (x) cos
    theta_x2 : dctn (both axes)                              # cos (x) cos

[measured: round-trip errors 4e-16; single-mode pickup at the stated slots.]

Ortho normalization detail (classic silent-bug site): on this grid the
orthonormal basis weights are sqrt(2/N) for sine frequencies 1..N-1 and cosine
frequencies 1..N-1, but sqrt(1/N) for the two exceptional slots (cosine m=0,
sine m=N). The derivative maps below transfer coefficients only within the
shared-weight range 1..N-1 and never touch the exceptional slots, so no
rescaling is ever needed. The cosine m=0 slot is legitimate DATA (the
x1-mean of theta, u2 -- e.g. mean stratification); the sine m=N slot is
structurally dropped (it lies inside the dealiasing kill zone anyway).

### 2.6 Derivative maps between bases

d/dx of sin(m x) = +m cos(m x); d/dx of cos(m x) = -m sin(m x). At equal
frequency and equal ortho weight the coefficient transfer is exact:

    def d_sin2cos(F, axis):          # d/dx along a sine axis
        G = zeros_like(F); m = 1..N-1
        G[slot m] = +m * F[slot m-1]     # cos slot 0 stays 0 (no mean is created)
        return G                          # sine freq N dropped

    def d_cos2sin(F, axis):          # d/dx along a cosine axis
        G = zeros_like(F); m = 1..N-1
        G[slot m-1] = -m * F[slot m]      # sine freq N slot stays 0
        return G

[measured: max error 1.9e-14 against analytic derivatives at N=16.]

### 2.7 Poisson solve

Both w and psi are sin(x)sin with frequency pairs (m,n), m,n >= 1:

    LAM[i,j] = (i+1)^2 + (j+1)^2          # = m^2 + n^2, eigenvalue of -Lap
    psi_hat  = - w_hat / LAM              # Lap(psi) = w

Never singular (no zero mode in a sine-sine basis). Velocities:

    u1_hat = - d_sin2cos(psi_hat, axis=1)     # u1 = -psi_x2, sin(x)cos
    u2_hat = + d_sin2cos(psi_hat, axis=0)     # u2 = +psi_x1, cos(x)sin

[measured: discrete divergence d(u1)/dx1 + d(u2)/dx2 = 1.1e-17.]

### 2.8 Dealiasing (2/3 rule on the parity bases)

The parity fields on [0,pi]^2 are restrictions of 2pi-periodic fields on a
2N-point grid per direction; the cell-centered N-grid is exactly half of that
2N periodic grid, and pointwise products of extensions equal extensions of
products (parities multiply). Hence the classic periodic dealiasing theory
applies verbatim with M = 2N points: a quadratic product of fields truncated
at frequency K is alias-free in the retained band iff 3K < 2N. Take

    K = (2*N - 1) // 3        # N=128 -> 85, 256 -> 170, 512 -> 341, 1024 -> 682

and zero every coefficient whose frequency (not slot!) exceeds K, per axis,
in: the prognostic fields entering the RHS, and the forward transform of each
nonlinear product. Grid-specific alias identities (for the record; the mask
kills all of it) on x_j = (j+1/2)pi/N:

    sin((2N-m)x_j) = +sin(m x_j)      cos((2N-m)x_j) = -cos(m x_j)
    cos(N x_j)     =  0 identically   (why DCT-II stops at N-1)

[measured: all three identities hold to 9e-15.]

With this mask the grid product equals the exact Galerkin projection of the
quadratic term, which is what makes the semi-discrete budget identities of
Section 4 EXACT (measured below), not merely small.

### 2.9 Quadrature exactness lemma

Midpoint quadrature Q[f] = h^2 * sum_j f(x_j) integrates exactly every trig
polynomial of the extension with cosine content of frequency < 2N (the first
aliasing line is m = 2N; [measured: quad error 2.6e-15 at m=2N-1, first
failure exactly at m=2N]). Products of two K-truncated fields have content
<= 2K < 2N, and even the cubic budget integrands (w * u.grad(w)) have content
<= 3K < 2N. Consequence: every discrete integral in Sections 4-6 is EXACT for
the represented fields -- budget residuals isolate time-integration error and
(if under-masked) aliasing, with zero quadrature noise.

### 2.10 RHS assembly per evaluation

Inverse-transform 6 coefficient arrays to the grid (u1, u2, w_x1, w_x2,
theta_x1, theta_x2), form the two advection products pointwise, forward-
transform each equation's product sum with that equation's class transform
(the whole RHS of (3) is sin(x)sin; of (4) is cos(x)sin), apply the mask, add
the buoyancy transfer theta_x1 (already sin(x)sin coefficients) and the
diffusion terms spectrally (-nu*LAM*w_hat, -kappa*LAM_theta*theta_hat with
LAM_theta[i,j] = i^2 + (j+1)^2). 8 two-dimensional transforms per RHS
evaluation, 32 per RK4 step, all O(N^2 log N). No dense operators anywhere.

## 3. Initial-condition family and the oracle theorem

### 3.1 Smooth corner-localized family, single dial A

Chen-Hou style: pure buoyancy data, zero initial vorticity (in the Luo-Hou
analogy: pure swirl, w^phi_0 = 0), theta0 >= 0 on the box, localized at the
corner (0,0), smooth with FINITE spectrum (trig polynomial, so the parity
extension is entire and the discrete representation is exact at any N above
the cutoff):

    bump_r(y) = ((1 + cos y)/2)^(r/2)          # r even: finite cosine series,
                                               # max frequency r/2, peak at 0,
                                               # width ~ 2/sqrt(r)
    theta0(x) = (A / g*) * bump_p(x1) * sin(x2) * bump_q(x2)
    w0(x)     = 0

with p, q even integers, defaults p = q = 32. The x2 profile sin(y)*bump_q(y)
vanishes at both walls (in-class), peaks at

    y* = arccos(q/(q+2))                       # q=32: y* = 0.34470...
    g* = sin(y*) * ((1+cos y*)/2)^(q/2)        # q=32: g* = 0.20958896544...

so the closed-form normalization makes sup(theta0) = A exactly (x1 factor
maxes to 1 at x1 = 0). Max frequencies: p/2 = 16 in x1, q/2 + 1 = 17 in x2 --
exactly representable for every N >= 32 and far inside every dealias mask.

Parity check: bump_p(x1) is even in x1 with a pure cosine series (even p) --
cos class; sin(x2)*bump_q(x2) is odd with a pure sine series -- sin class.
theta0 in cos(x)sin. w0 = 0 trivially in class.

Scenario narrative (signs verified in 1.2): the warm column on the axis just
above the wall rises; by the flank-torque mechanism this drives a dipole
(w < 0 for x1 > 0), whose near-wall flow converges TOWARD the corner along
the wall, steepening theta_x1 corner-adjacent -- the Luo-Hou hyperbolic-corner
sharpening loop. A is the single amplitude dial, the analog of the 1D
engine's A: the fate map (decay / sharpening cascade) is explored in A at
fixed shape (p, q).

### 3.2 The oracle theorem (intellectual-honesty section)

The proven theorem this scenario calibrates against:

Jiajie Chen & Thomas Y. Hou proved finite-time singularity formation from
SMOOTH initial data for the 2D inviscid Boussinesq equations in the presence
of a boundary, and correspondingly for 3D axisymmetric Euler in a cylinder:
a STABLE, NEARLY SELF-SIMILAR blowup at the boundary -- the rigorous
confirmation of the scenario Luo & Hou discovered numerically in 2014. The
proof is computer-assisted: construction of an approximate self-similar
profile plus rigorous, interval-arithmetic-verified stability estimates.
I am confident of: the authors, the equations, the essential role of the
boundary, stability of the blowup, near-self-similarity, smoothness of the
data. UNVERIFIED details -- check against the papers before any oracle-grade
claim: the exact domain of the smooth-data theorem (half-space vs strip vs
periodic-in-x1), the admissible data class and whether w0 = 0 with our
specific parity-box profile lies in the proven stable basin, arXiv
identifiers (believed 2210.07191 for the smooth-data work; the earlier
C^{1,alpha} boundary blowup result is separate), publication venues and
years. Luo & Hou: "Potentially singular solutions of the 3D axisymmetric
Euler equations", PNAS 2014 (volume/pages UNVERIFIED).

Dimensional scaling predictions used by the diagnostics (derived, not cited:
the inviscid equations admit the two-parameter scaling w -> mu*w(lambda x,
mu t), theta -> (mu^2/lambda)*theta(lambda x, mu t), so w carries dimension
1/time and grad(theta) dimension 1/time^2):

    sup|w|         ~  c /(T*-t)         modulo slow corrections
    sup|grad theta| ~ c'/(T*-t)^2

## 4. Budget identities for live monitors

All derived with the parity boundary facts: u.n = 0 on all sides, w = 0 on
all sides, theta = 0 on the horizontal walls, theta_x1 = 0 on the vertical
walls (sine factor). Every boundary term below vanishes identically.

### 4.1 Kinetic energy

E = int |u|^2/2. Using u.grad terms' flux form and int u.grad(p) = 0:

    dE/dt = int(theta * u2)  -  nu * int(w^2)                            (E)

(The viscous step uses Lap(u) = (-w_x2, +w_x1) for divergence-free u, then
int u.Lap(u) = -int(w^2) + boundary(w * dpsi/dn) and w = 0 on d(box).)
Buoyancy work int(theta*u2) > 0 when warm fluid rises: consistent with 1.2,
and the -nu*int(w^2) sign confirms the Section 1.3(a) viscosity fix.
[measured: discrete residual of (E) on a generic in-class state = 0.0 exactly
against the assembled RHS; buoyancy-work term reproduced.]

### 4.2 Scalar variance

    d/dt int(theta^2)/2 = - kappa * int|grad theta|^2                    (T)

(int theta u.grad theta = flux of theta^2/2 through walls = 0; diffusion
boundary term theta*dtheta/dn = 0: theta vanishes on horizontal walls,
dtheta/dn = theta_x1 vanishes on vertical walls.) Inviscid: theta^2 exactly
conserved -- gate G2. [measured: semi-discrete drift rate 9e-19 on an O(1)
state.]

### 4.3 Enstrophy

    d/dt int(w^2)/2 = int(w * theta_x1)  -  nu * int|grad w|^2           (Z)

(advective flux = 0; viscous boundary term w*dw/dn = 0 since w = 0 on
d(box).) NOT conserved inviscidly: int(w*theta_x1) is the production term --
this is the singularity engine, and (Z) is its live meter.
[measured: discrete residual of (Z) = 6.9e-17 at production 0.084.]

### 4.4 Total (kinetic + potential) energy -- bonus exact invariant

With potential energy P = -int(theta * x2):  dP/dt = -int(theta*u2)
(inviscid; boundary flux of x2*theta*u vanishes), hence

    d/dt [ E + P ] = 0        exactly, inviscid                          (H)

x2 here is a fixed quadrature weight, not a field; with kappa > 0, (H) gains
-kappa*[int x2*Lap(theta)] which carries a nonzero wall flux at x2 = pi --
inviscid runs get the exact invariant, viscous runs get the identity with the
flux term written out.

### 4.5 G3-mode wave energy (background stratification on)

With theta = B*x2 + theta' (background handled analytically, Section 5/G3):

    d/dt [ E + int(theta'^2)/(2B) ] = 0     exactly, inviscid            (W)

(dE/dt = int(theta'*u2) since the background does no net work:
int x2*psi_x1 = 0 by psi's x1-sine parity; and d/dt int theta'^2/2
= -B int(theta'*u2).)

### 4.6 Discrete caveat -- the meter, not a nuisance

In continuum all four identities are exact. Discretely, with the 2/3 mask the
nonlinear terms are exact Galerkin and every quadrature is exact (2.9), so
the SEMI-discrete identities are exact too [measured above at 1e-16..1e-18].
What remains in a running budget residual is therefore, in order:

1. RK4 time-integration error, O(dt^4) per unit time -- the M1-style step
   tripwire of audit.py: compare the realized change of E, theta^2, Z across
   a macro step against Simpson quadrature of the RHS inner products over the
   stage states; a bit-flip or NaN precursor jumps it by orders of magnitude.
2. Aliasing, ONLY if the mask is disabled or a bug lets tail content through
   -- the M2-style meter of audit.py: on smooth states the residual sits at
   roundoff; growth toward 1e-2 flags a triple-product resolution failure.
3. Roundoff accumulation.

A properly dealiased run whose budget residual scales as dt^4 and whose
spectral tail (Section 6) is clean is trustworthy; any other signature is a
diagnosis, not noise.

## 5. Gate suite (exact expected values and tolerances)

All gates fp64, deterministic, fixed dt (no adaptivity inside gates).
Gate constants live in one place in the code and are asserted, swarm_m1
style: a failed gate aborts the run before any science output.

### G1 -- theta = 0 sector: Taylor-Green exact decay

theta0 = 0, w0 = sin(x1)*sin(x2) (mode m = n = 1; any single sin-sin mode is
a Laplacian eigenfunction, u.grad(w) = J(psi,w) = 0 identically since
w = -2*psi). With theta = 0 the system is 2D Navier-Stokes; the mode is a
steady Euler solution and with viscosity decays in exact shape:

    w(x,t) = exp(-nu*(m^2+n^2)*t) * sin(m x1) * sin(n x2)

Run: N = 128, nu = 0.01, dt = 1e-3, 1000 steps to t = 1. Numbers to match:

    exp(-0.02) = 0.9801986733067553
    pointwise:  max_j | w_num(x_j, 1) - 0.9801986733067553*sin(x1_j)*sin(x2_j) |

Tolerance: < 1e-10 (tripwire). Expected ~1e-13: the discrete advection term
on this mode is pure roundoff cancellation [measured: 2.2e-14 at N=64], and
RK4 error on the linear decay is O((nu*2*dt)^5/step) ~ 1e-19. Additionally
assert max|theta| == 0.0 BYTE-EXACT at every sampled step (all theta-RHS
terms are proportional to theta; IEEE products with exact zeros stay exact
zeros -- any nonzero is a wiring bug, not an error).
Note: compare pointwise on the grid, not sup-vs-sup -- the cell-centered grid
max of sin*sin is cos^2(pi/2N), not 1 (see Section 6 grid-sup caveat).

### G2 -- inviscid theta^2 conservation under full nonlinear evolution

IC family of 3.1 at A = 4 (O(1) dynamics), nu = kappa = 0, N = 256
(K = 170), dt = 1e-3, t in [0,1]. Monitor
D(t) = |Q[theta^2](t) - Q[theta^2](0)| / Q[theta^2](0).

Semi-discrete conservation is EXACT (4.2 measured at 9e-19 instantaneous), so
D is pure RK4-accumulation + roundoff:

- Hard tripwire: D(1) < 1e-8.
- Scaling assertion: halving dt divides D(1) by 16 (accept factor in [8,32])
  until the roundoff floor (~1e-14) is reached.
- Expected: 1e-12 .. 1e-10 (ESTIMATE; promote the measured value to the
  tripwire constant after first light, per meter-era practice).

### G3 -- linear physics: internal gravity waves (dispersion relation)

Base state: theta_bg = +B*x2, B > 0 (STABLE under pinned conventions --
sign derivation in 1.3(b); the linear profile is not in the cos(x)sin class,
so the background is handled ANALYTICALLY: evolve (w, theta') with
theta = B*x2 + theta'; the only modification is one extra in-class term
-B*u2 in the theta'-equation, and the vorticity forcing stays theta'_x1).

Derivation (do-not-trust-the-prompt rerun): linearize about rest,
w'_t = theta'_x1, theta'_t = -B*u2' = -B*psi'_x1, Lap(psi') = w'. Single
in-class mode w' = a(t) sin(k1 x1) sin(k2 x2),
theta' = b(t) cos(k1 x1) sin(k2 x2):

    da/dt = -k1 * b,   db/dt = +B*k1/(k1^2+k2^2) * a
    =>  d2a/dt2 = -[B*k1^2/(k1^2+k2^2)] * a

    sigma^2 = B * k1^2 / |k|^2      (confirms the tasked formula; note
                                     sigma <= sqrt(B) for ALL modes)

Exact frequencies to match at B = 1:

    mode (1,1): sigma = 1/sqrt(2)  = 0.7071067811865476   T = 8.885765876316732
    mode (2,1): sigma = 2/sqrt(5)  = 0.8944271909999159   T = 7.024814731040727

Measurement protocol: N = 256, dt = 1e-3, FULL NONLINEAR code with the B
term on; IC: w'_hat[(1,1) slot] = eps = 1e-4, theta' = 0; record the (1,1)
sine-sine coefficient a(t) every step for 4 periods; frequency from
zero-crossing times (linear interpolation between samples),
sigma_meas = pi / mean(consecutive crossing gaps).

Tolerance: |sigma_meas - sigma| / sigma < 1e-8 (tripwire).
[measured: 7.7e-14 at N=64, dt=1e-3, eps=1e-4, 4 periods -- the single-mode
advection self-term vanishes identically (Taylor-Green structure), so
nonlinear contamination does not shift the frequency at O(eps).]
Repeat for mode (2,1). Cross-meter: the wave invariant (W) of 4.5 must hold
to the same dt^4 standard as G2.

### G4 -- resolution-doubling convergence order

(a) Spatial (spectral): scenario IC (A = 4), inviscid, fixed dt = 5e-4,
integrate to t = 0.5 (smooth pre-singular window) at N in {128, 256, 512}.
Error e_N = max|w_N - w_512| on the common coefficient set (compare
coefficients, not grids). Assert effective order log2(e_128/e_256) >= 8
(spectral accuracy makes this large; 8 is the tripwire floor), and
e_256 itself < 1e-8.
(b) Temporal (RK4): N = 256 fixed, dt in {2e-3, 1e-3, 5e-4} against a
dt = 1.25e-4 reference; order p = log2(e_dt/e_{dt/2}) must satisfy
p in [3.7, 4.3].

### G5 -- parity/symmetry preservation at roundoff, long integration

The code evolves reduced coefficient arrays, so parity cannot drift by
construction; what CAN break it is an index-shift or normalization bug in the
derivative maps. Test the discrete EQUIVARIANCE that such bugs destroy. The
continuum flow commutes with two involutions (verify by substitution into
(3)-(5)):

    S1: w -> -w(pi-x1, x2),  theta -> +theta(pi-x1, x2)
    S2: w -> -w(x1, pi-x2),  theta -> -theta(x1, pi-x2)

In coefficients, S1 flips the sign of every odd-m1 sine slot of w etc. --
exact, no interpolation. Protocol: N = 128, scenario IC at A = 4, dt = 1e-3,
1000 steps; evolve IC and S_i(IC); assert

    max| S_i(flow(IC)) - flow(S_i(IC)) | / max|w|  < 1e-12     (i = 1, 2)

Expected ~1e-14 (pure roundoff non-commutativity). Run S1 and S2.
Pre-gate (cheap, once at startup): transform round-trip and derivative-map
checks against analytic functions, err < 1e-13 [measured: 1.9e-14].

## 6. Singularity-approach diagnostics

D1  sup|w|(t): grid max of |w|, with argmax location. Grid-sup caveat: the
    cell-centered grid undersamples a max lying between points (factor
    cos^2(pi/2N) already for the smooth (1,1) mode); near-singular fields
    make this worse. Report grid-sup as the primary (honest) number; the
    trust wire D6 says when it stops being meaningful.
D2  sup|grad theta|(t): both derivatives spectrally, grid max of the norm.
D3  Growth-exponent extraction (post-processing, never in the physics path):
    for M(t) = sup|w| assumed ~ C*(T*-t)^(-gamma), the log-derivative ratio
        gamma_inst = (dL/dt)^2 / (d2L/dt2),  L = ln M
        T*_inst    = t + gamma_inst / (dL/dt)
    is exact for a pure power law and needs no prior T*. Evaluate by local
    quadratic fits of L on a uniformly resampled time series (the adaptive dt
    makes raw samples nonuniform). Oracle expectation: gamma -> 1 for sup|w|
    and gamma -> 2 for sup|grad theta| (dimensional scaling, 3.2), with slow
    corrections (nearly-self-similar, not exactly).
D4  Corner-localization measure: r_max(t) = |argmax(|w|)| (distance of the
    vorticity max from the corner), and enstrophy concentration fractions
        F_rho = Q[w^2 * 1_{|x|<rho}] / Q[w^2],   rho in {pi/8, pi/16, pi/32}
    with precomputed indicator masks. (Indicators break spectral quadrature
    exactness -- fine for a diagnostic, never used in a budget.)
D5  BKM integral: I(t) = int_0^t sup|w| dt', accumulated by trapezoid over
    accepted steps. Beale-Kato-Majda-type criterion: genuine blowup requires
    I -> infinity (classical BKM is for Euler; its Boussinesq variant's exact
    hypotheses UNVERIFIED -- track alongside D7 which is self-contained).
    Report I(t) and its growth rate; a saturating I with growing sup|w| is a
    numerics warning, not physics.
D6  Spectral-tail trust wire (constitutional, cf swarm_m1 TAIL_TRUST): with
    retained band f <= K, define per field
        tail = sum_{max(m,n) > 3K/4} |coef|^2 / sum |coef|^2
    for w and theta. tail > 1e-4: low-trust flag on every reported number
    from that time on. tail > 1e-2: resolution exhausted -- the approach is
    over, stop claiming physics, report the last trusted state. All fate/
    exponent claims carry the flag state, swarm-ledger style.
D7  Self-contained a priori bound (elementary, derived here): inviscidly,
    along particle paths Dw/Dt = theta_x1, so
        sup|w|(t) <= sup|w0| + int_0^t sup|theta_x1| dt'
    Both sides are measured independently; VIOLATION IS IMPOSSIBLE in exact
    arithmetic, so any measured violation is a hard numerics tripwire
    (the discrete analog of a conservation-law breach). Also monitor
    sup|theta|(t) <= sup|theta0| (advected scalar, max principle): the
    dealiased dynamics can overshoot pointwise at the truncation level
    (P_K is not max-principle-preserving), so its violation MAGNITUDE is
    another resolution meter, expected ~ tail size, not zero.

## 7. Time stepping

RK4 (classical, as in swarm_m1), coefficient-space state (w_hat, theta_hat).
Adaptive dt, a deterministic function of the current state only:

    dt = min( DT_MAX,
              C_a / (K * sup|u|),            # advective CFL, spectral form
              C_b / sqrt(B + sup|grad theta|),   # buoyancy/internal-wave limit
              C_v / (nu_max * 2*K^2) if nu or kappa > 0 )   # explicit diffusion

Rationale and constants:

- Advective: RK4's imaginary-axis stability reaches |lambda*dt| <= 2*sqrt(2)
  = 2.828; the largest advective eigenvalue magnitude is ~ K*sup|u|.
  C_a = 1.4 is a 2x margin. (Equivalently ~0.4 * h/sup|u| in grid units.)
- Buoyancy: linearization about the local theta gradient oscillates or grows
  at rate sigma with sigma^2 <= |grad theta| (G3 derivation: sigma^2 =
  gamma*k1^2/|k|^2 <= gamma locally); C_b = 0.5 resolves the fastest wave
  with ~12 stages per radian and respects the same RK4 bound with margin.
- Diffusive (viscous option only): RK4 real-axis limit 2.785; largest
  eigenvalue nu*2K^2. C_v = 1.4.
- DT_MAX = 5e-3 (quiet-state cap).
- Exhaustion guard: if dt < DT_MIN = 1e-9, terminate the approach and report
  the last trusted state with all meters -- no silent grinding.

sup|u| and sup|grad theta| are grid maxima of already-computed arrays: zero
extra transforms. dt depends on nothing but the state: bit-reproducible runs.

Laptop sizing (M1 Pro, 16 GB, fp64 scipy.fft -- ESTIMATES, calibrate at first
light and record in the run header):

- Gates: N = 128 / 256 as specified. Seconds to minutes each.
- Scenario default: N = 512 (K = 341). ~15 coefficient/grid arrays of 2 MB:
  memory trivial. Cost ~ 32 transforms/step at 2-6 ms each: ~0.1-0.2 s/step,
  ~1e4 steps/hour. ESTIMATE.
- Approach pushes: N = 1024 (K = 682), ~0.4-0.8 s/step ESTIMATE; use only
  after N = 512 runs place the interesting A window (1D-engine lesson: the
  swarm at moderate N brackets, the big-N run confirms).
- fp32/MPS batched mirror (swarm-style A-sweeps) is a later, separate build
  and must pass a cross-backend gate against this fp64 engine first.

## 8. Honesty ledger: deviations and open uncertainties

1. Free-slip realization. The parity box FORCES w = 0 and theta = 0 on the
   horizontal walls (2.3) -- equivalent to stress-free walls. Luo-Hou's
   inviscid cylinder constrains only u.n = 0: their wall vorticity and wall
   swirl are nonzero, and in the 3D analogy theta ~ (r*u^phi)^2 is MAXIMAL at
   the wall, while our theta vanishes there. Mitigation, not proof: the
   singular point already sits on vorticity zero-lines in the proven scenario
   (w odd across the symmetry plane vanishes AT the ring while sup|w| blows up
   beside it), so zero-lines through the corner do not preclude corner-
   adjacent blowup; but whether the parity-box data class lies in the
   Chen-Hou stable basin is UNVERIFIED and must be pinned against the paper
   before oracle-grade calibration claims. The box problem is, regardless,
   a self-consistent 2D Boussinesq scenario in its own right.
2. Citation precision. Names/structure of the Chen-Hou and Luo-Hou results:
   confident. arXiv IDs, venues, exact domains, data classes: UNVERIFIED
   (Section 3.2). This is the oracle; verify before calibrating against it.
3. Two sign fixes applied to the task statement (viscosity sign convention,
   stratification sign in G3), each cross-checked against an energy identity
   or a stability derivation: Section 1.3. If a different convention was
   intended upstream, these are the two lines to revisit.
4. w0 = 0 initial data matches the Luo-Hou pure-swirl analogy but its
   membership in the proven stable-blowup basin is UNVERIFIED; the A-dial
   sweep is precisely the experiment.
5. All laptop cost numbers in Section 7 are ESTIMATES pending first light.
6. G2/G3 expected values quoted from the N=64 verification harness; gate
   constants get promoted from the first full-engine measurements (meter-era
   practice: measure, then tripwire).

## 9. File plan (all under ~/parzival/boussinesq/, nothing outside touched)

    FORMULATION.md   this document
    engine.py        transforms, derivative maps, Poisson, RHS, RK4, dt law
    gates.py         G1-G5, asserts with the Section 5 constants
    monitors.py      Section 4 budgets + Section 6 diagnostics (M1/M2-style)
    scenario.py      IC family, A-dial runs, run-header with calibrations
    runs/            local run outputs (kept inside boussinesq/)

## Corrections (era B1 -> B2, 2026-07-22, oracle-verification round)

1. **S3.1 mis-attribution**: "Chen-Hou style: pure buoyancy data, zero initial
   vorticity" is wrong -- w0=0 is the LUO-HOU numerical IC (SIAM MMS 12(4)
   2014, eq 2.3a). The Chen-Hou PROVEN basin (arXiv 2210.07191 Thm 3) is
   profile-adjacent data within E* = 5e-6 of a nontrivial (w-bar, theta-bar)
   self-similar profile in a weighted norm; w0 = 0 fails by an enormous margin.
2. **S8.1 upgraded from risk to exclusion**: the free-slip parity box (theta,
   w odd across the wall) is the exact 4-fold-symmetry, no-boundary class
   Chen-Hou analyze and REJECT (arXiv 1910.00173 S1.4; global existence for
   their model problem in this class, Appendix A.8). theta odd in y forces
   |theta| <~ |y|, and wall-normal advection of theta_x destabilizes the
   focusing mechanism. The proven scenario REQUIRES theta nonzero on the wall,
   growing quadratically along it from the corner, and w nonzero on the wall.
   Era-B2 engine therefore needs a genuine boundary direction: x1 parity
   (theta even, w odd -- our x1 treatment already matches the theorem) times
   a true wall-normal discretization (Chebyshev with no-flow psi(x,0)=0).
   Our B1 ladder verdict (decelerating growth at every N) is CONSISTENT WITH
   the literature's global-existence expectation for this symmetry class --
   a blind reproduction, not a failure.
3. **D5 continuation criterion corrected**: whether vorticity-only BKM applies
   to inviscid 2D Boussinesq is OPEN (Elgindi-Jeong, Ann. PDE 6, 2020, S1.5).
   Rigor-backed criteria: int sup|grad u| or int sup|grad theta| (Chae-Nam
   1997/1999; Elgindi-Jeong Thm A). Primary meter: int sup|grad theta| dt.
   In the proven scenario int sup|w| diverges only LOGARITHMICALLY -- its
   apparent saturation must never be read as evidence against blowup.
4. **Pre-registered oracle exponents** (Chen-Hou II arXiv 2305.05660, MMS
   2025; Luo-Hou 2048^2): sup|w| ~ (T-t)^-1 (measured -0.9972);
   sup|grad theta| ~ (T-t)^-2; spatial collapse l(t) ~ (T-t)^2.92 (c_l/c_w
   ~ -2.92; measured 2.9133); velocity bounded, ~ (T-t)^0.46 at the
   stagnation point. Do NOT import Luo-Hou's 2.4568 (3D vorticity VECTOR,
   dominated by omega^r, omega^z; the Boussinesq analog is the -1 exponent).
5. **Citations**: all previously-UNVERIFIED references now pinned -- see the
   oracle verification report (vault: bsq-verify-round2).
