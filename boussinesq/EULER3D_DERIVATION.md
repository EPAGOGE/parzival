# The 3D corrections, derived, and why the homotopy readout is a slope not an endpoint

Working note, 2026-07-28. Route 4 of the euler3d-corrections derivation (my own hand
pass, independent of the three workflow routes).

## 1. The convention, pinned on three terms

The coded residuals carry an overall sign and a set of divided-variable multipliers
that must be reproduced exactly or every correction is wrong. Substitutions:

    xi = ln(1+rho),  rho = e^xi - 1,  G1 = rho*e^-xi/xi,  E1 = e^(a0*xi)/G1,  mu = 2+a0
    Omega = xi*A*e^(a0*xi),   theta = xi^2*B*e^((1+2a0)*xi),   Psi = xi^2*P*e^(mu*xi)
    x_phys = rho*cos(beta)   (3D AXIAL z),   y_phys = rho*sin(beta)   (wall-normal 1-R)

Pushing the substitution through the coded advection bracket:

    E1*(-LPp*A_b + P_b*LAa) = [e^(-a0*xi)/(rho*xi)] * [Psi_b*Om_rho - Psi_rho*Om_b]
                            = M_O * (-(u.grad Omega)),   M_O = e^(-a0*xi)/xi

with u = grad-perp Psi = (-Psi_y, Psi_x). Two independent confirmations of M_O:

  * the c_l term.  M_O*rho*d_rho(Omega) = G1*LAa, and the code has -G1*LAa. Matches.
  * the forcing.   M_O*d_x(theta) = E1*(G1*cos(b)*LB2b - sin(b)*B_b). Matches.

Same route gives M_B = e^(-(1+2a0)*xi)/xi^2 and M_P = G1^2*e^(-a0*xi), the latter
confirmed twice (M_P*Omega = xi*G1^2*A, and M_P*(1/rho^2)*Psi_bb = P_bb).

So a source term F in "d_t f + u.grad f = F" enters the coded residual as +M*F, and
the code's overall residual sign is minus the textbook one.

Two identities do all the work in what follows:

    xi*e^((1+a0)*xi)       = rho*E1
    xi^2*e^((2+a0)*xi)/rho = rho*E1/G1

## 2. The four corrections

3D axisymmetric Euler in Hou-Luo variables, with R the cylindrical radius, wall at
R = 1, and R = 1 - y_phys:

    u^x = u^z = 2*Psi - R*Psi_y  =  -Psi_y + (2*Psi + y*Psi_y)
    u^y = -u^R = R*Psi_x         =   Psi_x + (-y*Psi_x)

so the velocity correction is  du = (2*Psi + y*Psi_y,  -y*Psi_x)  and

    du.grad f = 2*Psi*f_x - y*(u.grad f)

The y-part is just -y times the advection already coded, which is why the corrections
reuse the existing brackets. With ADV_O = E1*(-LPp*A_b + P_b*LAa) and
ADV_B = E1*(-LPp*B_b + P_b*LB2b):

    RO += tau*( -2*rho*E1*P*(cosb*LAa  - (sinb/G1)*A_b) - rho*sinb*ADV_O )
    RB += tau*( -2*rho*E1*P*(cosb*LB2b - (sinb/G1)*B_b) - rho*sinb*ADV_B
                + 4*B*rho*E1*(cosb*LPp - (sinb/G1)*P_b) )
    RP += tau*( -(3.0/R)*rho*(G1*sinb*LPp + cosb*P_b) ),     R = 1 - rho*sinb

The recurring shape (cosb*L_f - (sinb/G1)*f_b) is the divided-variable d_x operator,
and it appears identically in all four places, which is a structural check on the
algebra rather than a coincidence.

CORRECTION to the stage-0 script: it recorded C2 as +y*Psi_x. It is -y*Psi_x. Degrees
are unaffected so the stage-0 verdict stands, but the sign was wrong on the page.

## 3. Why tau = 1 is not the answer

Every correction is O(rho) against an O(1) partner. That was the stage-0 test and it
passed. But rho = e^xi - 1 reaches e^25 ~ 7.2e10 at XMAX = 25, so R = 1 - rho*sinb
goes enormously negative in the far field and the corrections are not small there at
all. The degree count in xi says nothing about this, because the smallness is in a
DIFFERENT parameter.

That parameter is the blowup scale. Self-similar blowup shrinks the spatial scale like
eps = (T-t)^lambda, so the physical wall distance is 1 - R = eps*y_phys, giving
R = 1 - eps*y. Redoing the four corrections with eps present:

    C1  3/R * d_R Psi   vs   Laplacian:  d_R ~ eps^-1, Laplacian ~ eps^-2   -> O(eps)
    C2  (R-1)*d_z Psi   vs   d_z Psi                                        -> O(eps)
    C3  2*Psi           vs   R*d_R Psi ~ eps^-1 * Psi                       -> O(eps)
    C4  4*theta*d_z Psi vs   u.grad theta ~ eps^-2 * Psi*theta              -> O(eps)

All four vanish like eps. That is not an accident of this formulation, it IS the
Hou-Luo statement that 3D axisymmetric Euler near the wall is asymptotically 2D
Boussinesq, expressed as a scaling.

Consequence, and it retires the plan as written: in the exact self-similar limit
alpha_3D = alpha_2D identically. A homotopy driven to tau = 1 is not solving the 3D
self-similar problem; it is solving a fixed finite-time snapshot whose alpha is not a
self-similar exponent. Stage 2's fold hunt and stage 3's probe battery were both
designed against the wrong endpoint.

## 4. What replaces it

The corrections are exactly the leading correction to the self-similar ansatz, which is
the content of "nearly self-similar" in Chen-Hou. The finite, well-defined, and as far
as the record shows unmeasured quantity is therefore not alpha(tau=1) but

    d alpha / d eps  at eps = 0

the first-order sensitivity of the Boussinesq exponent to genuine 3D structure. It is
one linear solve against the Jacobian already factored at the certified 2D root:
differentiate the bordered system, put the correction operator on the right-hand side,
read the alpha component. No continuation, no fold, no ladder.

It also removes the R blowup: to first order 1/R = 1 + eps*y + O(eps^2), so nothing
divides by a quantity that changes sign, and the operator is linear in the corrections
rather than nonlinear in tau.

The derivations in section 2 are not wasted. The first-order operator is built from
exactly those four terms; only the readout and the cost change.

## 5. The adjudicated state (2026-07-28, three-lens adversarial pass, all computed)

The sensitivity was measured (adjoint route, exact to 1e-13 against direct), swept
with a cutoff, and the divergence exponent p was put through a seven-config axis
battery and a three-skeptic adjudication. Where it landed:

**p = 0.648 +- 0.016, grid-converged** across outer degree 12-24, XMAX 20-30,
Nb 28-36, mid-panel refinement, and seven fresh-seed Newton converges. The
identification p = 1 + alpha = 0.6553 survived adjudication with its mechanism
CORRECTED:

  * my transport-transpose balance was the wrong pairing. C ~ rho/xi^2 is the
    POISSON-row correction, and what multiplies it is the elliptic-block adjoint
      w_P = xi^2 * rho^alpha * sin(nu1*(b - eps_b)),   nu1 = pi/(pi/2 - 2 eps_b),
    measured with angular correlation +1.00000 against the wedge fundamental. The
    rho^alpha decay is the elliptic adjoint exponent mu - nu1 with mu = 2 + alpha:
    p = 1 + alpha traces to two exact structural constants, the Poisson substitution
    weight and the wedge fundamental, and the logs cancel IDENTICALLY in the shell
    density w_P * C_P = rho^(1+alpha). Measured P-block slope 0.660; the refused
    transport mechanism governs the O-block, predicted 1+2alpha = 0.311, measured
    0.322, carrying 1e-4 of the answer.

**The far-field log structure holds pointwise** (kappa = 1.000 +- 0.001 on 30/35
beta columns; B and P at kappa = 2 cleanly), with two evidentiary repairs on the
record: the forcing/orphan crossover sits INSIDE the fit window at xi* = 7.6, and
envelope fits are contaminated by the near-axis columns where the rho^alpha
correction amplitude reaches -6.9. Surviving form:
A = (f(beta)/xi)(1 + c(beta) rho^alpha + ...).

**The drift law survives as a scaling law only.** Adjudicated form:

    alpha_3D(t) - alpha_2D  =  C * (T-t)^(1.0 +- 0.1)

The EXPONENT is cutoff-scheme-robust (the physical wall-distance cutoff rho*sinb
gives composite slope 0.3497 vs -alpha = 0.3447, tighter than the rho scheme) and
convention-robust across lambda = -1/alpha vs lambda = c_l. The CONSTANT is not
computable from the inner region: 73-80% of the truncated integral sits in shells
where the linearization is O(1) wrong; the 3/R geometric series contributes the same
eps-power at every order (resummation shifts the amplitude ~2x); the amplitude is
scheme-dependent (1.6-2.4x); the outer region contributes at the same power
(saturated-kernel slope -0.3482 vs p-1 = -0.3471) plus an axis log-layer worth
0.35-0.70x the whole linear answer. One linear solve at the 2D root fixes the
exponent; only the true 3D outer/axis problem fixes the coefficient.

**What a time-march would test** (G4b design): the prediction is the exponent 1.0
+- 0.1 in (T-t), not the amplitude. A march that fits alpha(t) - alpha_2D against
(T-t) tests the whole chain: the O(eps) correction structure, p = 1 + alpha, and
the matching-scale argument, without needing the coefficient we cannot supply.

Side-discovery (battery, tension #40): refining the mid panels (16,40)->(20,48)
moved h_id from -1.06e-3 to +1.56e-4, a 7x shrink AND sign crossing -- first direct
observation of the corner defect vanishing along the resolution axis.

## 6. The derivation workflow's verdict (10 agents, 3 routes + round trips + refuters)

The section-2 expressions are CERTIFIED as algebra and KILLED as a homotopy. Both
verdicts are load-bearing.

**Certified.** Four independent derivations byte-agree (the three workflow routes
plus this note's hand pass; the recombined forms match to 1.96e-16 on the live
grid). Closure is exact: dRO and dRB equal (full 3D residual) minus (coded 2D)
as sympy zeros, so nothing is missing and nothing is double-counted. The sign
convention was pinned from the code by discrimination, not assumption: only
u = (-Psi_y, +Psi_x) with residuals M*(RHS-LHS) closes, every flip fails, and a
corner-pin parity argument fixes the map orientation. A 20-variant mutation battery
rejected every corruption. tau = 0 recovery is BITWISE on all 9398 residual rows.
One adjudication went AGAINST the 2-1 route majority: tau belongs INSIDE
R = 1 - tau*rho*sin(beta), and tau is not an abstract dial -- it IS lambda, the
blowup-length to cylinder-radius ratio. The canonical patch is
euler3d_corrections_patch.py (self-contained, NOT installed).
d/dtau at 0 = -3*rho*ByP exactly, so the eps-linear sensitivity operator used in
sections 3-5 is the derivative of the adjudicated form.

**Killed, twice, on axes the certification does not touch.**

*Corner degrees (serious).* Section 2's "one degree subleading" holds for five of
six terms and FAILS for C1 on the solution manifold: RP's degree-0 partner
self-annihilates via the corner ODE (P_bb + 4P = 0 with the known sin(2b)
structure), so the effective partner drops to degree 1, where C1 lives. Relative
order 0, measured both ways (breaking the corner ODE restores C1's nominal +1).
C1's leading corner action is a GAUGE RENORMALIZATION, WX -> WX - 6*tau*c with
6c/WX = 6.2126, landing at corner order 1 -- exactly where alpha is first selected
(order 0 is alpha-blind and closes as h_id = 0 identically). Consequences: a
wall-layer-windowed 3D readout would measure gauge motion, not alpha; and h_id's
3D analogue is not h_id with the same constants. The stage-0 degree screen passed
because it ran on generic fields -- the sixth instance of the campaign's recurring
correction shape (the checked axis fine, the content on an unrepresented axis).

*Limits (serious).* Only 463/3132 grid nodes sit at y_phys < 1, inside the
cylinder the map describes. For every tau that keeps R > 0 everywhere
(tau < 1.39e-11 at XMAX=25), the correction inside the valid region sits at the
Newton tolerance floor: the perturbative window is EMPTY of signal where the map
is valid. The raw d/deps RHS grows like exp(0.90*XMAX) with no XMAX limit.
So the raw unwindowed dalpha/deps scalar means nothing on this grid; the windowed
far-field EXPONENT (sections 4-5) is the only meaningful readout, and no
coefficient C -- not even a scheme-dependent one -- is quotable from the profile.

Net state of the 3D program: six certified correction expressions, an exponent-only
drift law alpha_3D - alpha_2D ~ (T-t)^(1.0 +- 0.1) with p = 1 + alpha carried by
the elliptic adjoint, an empty homotopy window retiring any tau-solve on this grid,
and two live corner facts (the WX renormalization; h_id's tau-dependence) that any
future 3D corner identity must carry. The coefficient belongs to the outer/axis
problem (tension #38), which the profile does not contain.
