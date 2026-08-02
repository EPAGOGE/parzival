# Theorems: geometric viscous regularization and the moving equation

2026-07-30, late. Companion to SIGMA_LAMBDA.md. Rigor labels are explicit per
statement: PROVEN (complete proof below), PROVEN GIVEN (Hi) (complete proof
from stated hypotheses), CITED (literature), MEASURED (this campaign).

Throughout: u a smooth solution of 3D incompressible Navier-Stokes with
viscosity nu > 0 and finite energy E_0 = (1/2)||u_0||_{L2}^2 on [0,T),
omega = curl u, xi = omega/|omega| where omega != 0, S = (grad u + grad u^T)/2,
alpha = xi . S xi the stretching rate, P(t) = { |omega| >= (1/2)||omega||_inf }.

---

## Theorem 0 (Batchelor cap on the direction gradient). PROVEN.

Let t be a time at which the vorticity maximum is attained at an interior
point and the upper Dini derivative of ||omega(t)||_inf is nonnegative (the
peak is not strictly decaying). Then at some maximizing point x_m,

    nu |grad xi(x_m)|^2  <=  alpha(x_m)  <=  lambda_max(S(x_m)),

hence

    |grad xi(x_m)|  <=  sqrt( lambda_max(S(x_m)) / nu ).

Proof. Wherever omega != 0, write omega = |omega| xi. Then
Delta omega = (Delta|omega|) xi + 2 (grad|omega| . grad) xi + |omega| Delta xi.
Dot with xi. Since |xi|^2 = 1, xi . d_i xi = 0 for each i, so the middle term
vanishes, and xi . Delta xi = d_i(xi . d_i xi) - |grad xi|^2 = -|grad xi|^2.
Therefore the pointwise identity

    xi . Delta omega = Delta|omega| - |omega| |grad xi|^2.          (I)

The vorticity equation d_t omega + u.grad omega = omega.grad u + nu Delta omega
dotted with xi gives, using xi.(omega.grad u) = |omega| alpha and (I),

    (d_t + u.grad)|omega| = alpha |omega| + nu Delta|omega| - nu |grad xi|^2 |omega|.

At an interior spatial maximum x_m: grad|omega|(x_m) = 0 and
Delta|omega|(x_m) <= 0. The upper Dini derivative of the maximum of a smooth
family is bounded by the maximum of the pointwise derivatives over the argmax
set (Danskin), so

    d+/dt ||omega||_inf <= ( alpha(x_m) - nu |grad xi(x_m)|^2 ) ||omega||_inf

for some maximizer x_m. Nonnegativity of the left side forces
nu |grad xi(x_m)|^2 <= alpha(x_m), and alpha = xi.S.xi <= lambda_max(S). QED.

Corollary 0.1 (sigma cap). PROVEN GIVEN (A). Assume (A): along a sequence of
growing-peak times approaching T, lambda_max(S(x_m)) <= K |omega(x_m)|
(strain comparable to vorticity at the peak; measurable, and standard in
practice, though Calderon-Zygmund alone gives it only up to a logarithm).
Then the peak-local invariant Lambda_pt = |grad xi(x_m)| |omega(x_m)|^{-1/2}
obeys

    Lambda_pt <= sqrt(K/nu)     independently of ||omega||_inf,

so sigma_pt = dln Lambda_pt / dln ||omega|| <= 0 along growth. Any nu > 0
caps the geometry at Batchelor; only nu = 0 permits the measured inviscid
sigma = +1.

Confrontation with data. MEASURED: inviscid corner flow sigma = +1.01 (triple
validated). [RETRACTED 2026-08-02: the viscous ladder values below are
window-averaged REGIME MIXTURES -- sigma(A) crosses over from inviscid-like
to deeply depleted inside every trusted window (M4_SIGMA_SPREAD.out
half-split diagnostic). Certified replacement: deep-collapse sigma_Lambda =
-1.12/-1.24 at nu=1e-4 and -1.25/-1.29 at nu=1e-3, cross-grid spreads
0.121/0.032 (M4_SIGMA_DEEP.out). The sign conclusion below survives; the
magnitudes do not.] Viscous ladder sigma_PEAK = -0.37, -0.37, -0.43, -0.54,
-0.57 at nu = 1e-5 .. 1e-3: all <= 0 as the corollary requires, sitting
below the cap (dissipation, plus the peak is not always growing). The
observed discontinuity of sigma at nu = 0+ is exactly the content of
Theorem 0 [demoted 2026-08-02 to Corollary 0 of Constantin's
|omega|-equation, NOVELTY.md C3]: the cap exists for every nu > 0 and is
absent at nu = 0. The mechanism is proven; the ladder shows it saturating.

---

## Theorem 1 (conditional exclusion of type-I blowup by direction decay).
## PROVEN GIVEN (H1), (H2), (H3).

Hypotheses, for t in some interval (T-delta, T):

(H1) Type-I frame: c/(T-t) <= ||omega(t)||_inf <= M/(T-t).

(H2) Coherence: there is lambda_0 >= 1 such that for every x in P(t) and
     every y,
     |sin angle( xi(x+y,t), xi(x,t) )| <= min( 1, lambda_0 Lambda(t)
     ||omega(t)||_inf^{1/2} |y| ).
     (Within the direction coherence length of a peak point, the modulus of
     continuity of xi is controlled by its gradient scale. This is the
     definition of a coherent peak structure; it is checkable on data and is
     the sole structural hypothesis.)

(H3) Direction decay: Lambda(t) <= C_L ||omega(t)||_inf^{sigma} with
     sigma < -1/2.

Conclusion: T is not a singular time.

Proof. Step 1 (Constantin's identity). For divergence-free u with the decay
our finite-energy smooth setting provides,

    alpha(x) = (3/4pi) P.V. int D( yhat, xi(x+y), xi(x) ) |omega(x+y)|
               dy / |y|^3,

with |D| <= |sin angle(xi(x+y), xi(x))|. (Constantin 1994; the kernel
vanishes with alignment. This is the structure destroyed by Tao-type
averaging, which is why the argument is not barred by the averaged
counterexample.)

Step 2 (splitting). Fix x in P(t), rho > 0. Using (H2) on |y| <= rho and
|sin| <= 1 with Cauchy-Schwarz on |y| > rho:

    near:  (3/4pi) lambda_0 Lambda ||omega||^{1/2} ||omega||_inf
           int_{|y|<=rho} |y|^{-2} dy  = 3 lambda_0 Lambda ||omega||^{3/2} rho,
    far:   (3/4pi) ||omega||_{L2} ( int_{|y|>rho} |y|^{-6} dy )^{1/2}
           = c_1 ||omega||_{L2} rho^{-3/2},  c_1 = (3/(4pi))^{1/2}.

Minimizing A rho + B rho^{-3/2} over rho gives c_2 A^{3/5} B^{2/5}, so

    sup_{P(t)} alpha  <=  C_0 lambda_0^{3/5}
        ( Lambda ||omega||_inf^{3/2} )^{3/5} ||omega||_{L2}^{2/5},   (II)

with C_0 absolute.

Step 3 (energy budget). The energy identity gives
nu int_0^T ||omega||_{L2}^2 dt <= E_0.

Step 4 (Gronwall and Holder). The argmax set lies in P(t), so by the Dini
estimate of Theorem 0's proof (discarding the helpful viscous term),
||omega(t)||_inf <= ||omega(t_0)||_inf exp( int_{t_0}^t abar ), where
abar(s) = sup_{P(s)} alpha. Triple Holder with exponents (5/3, 5, 5) on (II):

    int_t^T abar ds <= C_0 lambda_0^{3/5}
        ( int_t^T Lambda ||omega||_inf^{3/2} ds )^{3/5}
        ( E_0/nu )^{1/5} ( T-t )^{1/5}.

Step 5 (the trigger). By (H3) and (H1),
Lambda ||omega||^{3/2} <= C_L M^{sigma+3/2} (T-s)^{-(sigma+3/2)}, and
sigma + 3/2 < 1, so the integral converges. Hence int^T abar < infty, hence
||omega||_inf is bounded up to T, hence by Beale-Kato-Majda (valid for
Navier-Stokes) T is not singular. QED.

Remarks. (i) For blowup rate (T-t)^{-gamma}, gamma > 1, the same chain
requires sigma < 1/gamma - 3/2: faster blowup demands even more alignment
failure. (ii) The theorem consumes exactly three ingredients: kernel
geometry, energy, and the measured-class object Lambda. No other structure.

---

## The gap, stated exactly.

Theorem 0 proves viscosity forces sigma_pt <= 0 at growing peaks.
Theorem 1 needs sigma < -1/2 to close regularity for type-I.
The entire remaining mathematical distance on this route is the strip

    -1/2 <= sigma <= 0.

MEASURED [SUPERSEDED 2026-08-02: the -0.37..-0.57 ladder is retracted as a
window-average regime mixture; the certified deep-collapse values are
-1.12..-1.29 at nu in {1e-4, 1e-3}, BELOW -1/2 at both viscosities on both
grids (M4_SIGMA_DEEP.out). The "crossing -1/2 by nu = 3e-4" claim is void;
the certified measurement is already below -1/2 at both viscosities tried.]
The open problem is now sharp: prove that for Navier-Stokes the direction
sheet coarsens strictly faster than Batchelor, Lambda <~ ||omega||^{-1/2-eps},
or exhibit a solution class that rides the cap.

---

## Theorem 2 (the moving equation, formalized).

2a (endless calculation). PROVEN GIVEN the collapse law. With the derived
physical exponent c_l = -1/alpha = 2.9206 (far-field matching; branch family
accumulates at alpha_inf = -0.4722, all members c_l in [2.12, 2.92]),
following the collapse to vorticity level Omega on any uniform grid costs
degrees of freedom N ~ Omega^{2 c_l} in the meridional plane, so work per
decade of Omega multiplies by 10^{2 c_l} ~ 7 x 10^5. Total work to the
singularity diverges as a power, not a log. The calculation is endless in
the precise sense that no finite computation reaches it by marching.

2b (mode replacement). PROVEN GIVEN the collapse law. The active band
k*(t) ~ l(t)^{-1} ~ (T-t)^{-c_l} carries the growth. Two times have disjoint
(factor-2 separated) active bands as soon as (T-t')/(T-t) <= 2^{-1/c_l}
= 0.789. Every 21 percent step toward T, the set of modes doing the work is
completely replaced. Same law, brand new unknowns, at every scale, forever.
This is the literal form of the intuition: the equation does not change; its
working parts are new at every moment, and the previously controlled parts
retire without transmitting control (supercriticality: available control per
octave degrades like 2^{-1/2}).

2c (no stationary frame). CITED. Necas-Ruzicka-Sverak 1996 and Tsai 1998:
Navier-Stokes admits no exact backward self-similar blowup with finite
(or local) energy. If NS blows up at all, it blows up with NO stationary
rescaling frame in the exact sense. The surviving candidates are exactly the
moving-equation ones: discretely self-similar (stationary only
stroboscopically) or wandering (no periodic frame at all). The intuition is
not a heuristic here; it is the only unexcluded shape of the phenomenon.

2d (the knife edge). PROVEN GIVEN Theorem 0 + (A). Any type-I DSS or
asymptotically self-similar NS singularity has Lambda_pt log-periodic and
bounded, hence sigma_pt = 0 on period average, while Theorem 0 caps
sigma_pt <= 0 at growing instants. Therefore such a singularity must ride
the Batchelor cap:

    Lambda_pt(t) = Theta( sqrt( lambda_max(S(x_m)) / (nu |omega(x_m)|) ) ),

with log-periodic ripple, for all t near T. A DSS Navier-Stokes singularity
is geometrically MARGINAL: its direction sheet sits permanently at the
viscous equilibrium thickness, never finer, never coarser. Falsifiable
target: in any candidate DSS solution, nu^{1/2} Lambda_pt must be Theta(1)
with log-periodic modulation. Anything else kills the candidate.

---

## What the two heads look like after tonight.

Regularity head: close the strip. Show sigma <= -1/2 - eps for NS, i.e.
direction sheets coarsen strictly faster than Batchelor. Theorem 1 then
finishes type-I. All machinery on this route survives the averaging barrier
by construction.

Blowup head: live exactly on the knife edge. The candidate is a DSS solution
with nu^{1/2} Lambda_pt = Theta(1). The corner mechanism, whose inviscid
geometry is maximally non-marginal (sigma = +1), measured tonight as
inverting under any nu, is not that candidate. Whatever is, if anything is,
must be born marginal.

Status ledger: Theorem 0 unconditional (interior max, Dini). Corollary 0.1
and 2d conditional on (A). Theorem 1 conditional on (H1) type-I, (H2)
coherence, (H3) the trigger. 2a, 2b conditional on the collapse law
(derived + branch-closed). 2c cited. Every hypothesis is either standard,
checkable on data we hold, or named as the open frontier.
