# Kernel-hat architecture: the far-field kernels converted into the solver

Stage 1 of the kernel-conversion inoculation, derivation phase, 2026-07-28.
All identities below verified exact in sympy this session. Plan-time unspanned
axes minted as tension #44 BEFORE building (first application of the adjudicated
plan-time-minting rule).

## 1. Variables and the exact bundle collapse

Hatted (kernel-divided) fields, regular at BOTH ends:

    Ahat = xi * A        ->  f(beta)   as xi -> inf     (plateau, measured 4.85)
    Bhat = xi^2 * B      ->  g(beta)                    (3.80)
    Phat = xi^2 * P      ->  h(beta)                    (1.50)

The coded L-bundles collapse EXACTLY (sympy zeros, no approximation):

    LAa  = Ahat_x + a0*Ahat                       [all 1/xi terms cancel]
    LB2b = (Bhat_x + (1+2a0)*Bhat) / xi
    LPp  = (Phat_x + mu*Phat) / xi

With s(xi) = 1 - e^(-xi)  (= xi*G1 exactly), the hatted transport row (xi * RO):

    ROhat = (E1/xi) * [ -(Phat_x + mu*Phat)*Ahat_b / xi + Phat_b * LhA ]
            + (E1) * [ G1 * cosb * (Bhat_x+(1+2a0)Bhat)/xi ... forcing, same recipe ]
            - cl * s * LhA  +  cw * Ahat,        LhA = Ahat_x + a0*Ahat

Every coefficient is finite on [0, inf): E1/xi = e^(a0 xi)/s -> 0, s -> 1.

## 2. The two ends, same medicine

**At infinity the transport rows are vacuous** (verified: plateau limit = 0 using
cl*a0 = cw exactly): outflow needs no condition, so the endpoint row 0=0 is the
mirror of the corner dust. The fix is the corner fix: on the compactified panel
xi = xi2 + L*t/(1-t), DIVIDE the transport rows by (1-t)^2. The divided endpoint
row imposes Ahat_t = 0 (exactly true: the correction is c*e^(alpha*xi), which
beats the (1-t)^-2 pole). Stage-0 measured the correction structure flat at
5.19-5.78 after e^(-alpha*xi) compensation, so this is the right order, not a guess.

**The elliptic row SURVIVES at infinity** (verified limit, P_bb term restored):

    Phat_bb + mu^2 * Phat + Ahat = 0     at xi = inf

nondegenerate, determines h(beta) from f(beta). Note the structure: the corner
carries P_bb + 4P (wedge fundamental sin 2b), infinity carries Phat_bb + mu^2 Phat
with mu = 2 + alpha: the angular operators at the two ends are alpha-shifted, and
nu = mu at infinity is exactly the elliptic-adjoint decay identity (mu - 2 = alpha)
that carries p = 1 + alpha. The kernel places the similarity constant into the
boundary operator itself, which is the 1-to-1 conversion the proposal asked for.

## 3. Build recipe (stage-1 implementation, handoff)

New class in a NEW file (nothing in polar_cornerreg touched):

  1. Panels 0..K-2 unchanged, standard variables. Replace the LAST panel with the
     compactified panel: Chebyshev nodes in t on [0,1] INCLUDING t=1;
     xi(t) = xi2 + L*t/(1-t); d/dxi = ((1-t)^2/L) d/dt (chain second derivative
     for the elliptic block). Unknowns on this panel: Ahat, Bhat, Phat nodal values;
     t=1 values ARE f(beta), g(beta), h(beta).
  2. Rows on the compactified panel: hatted residuals of section 1; transport rows
     divided by (1-t)^2; elliptic endpoint row = the section-2 limit row.
  3. INTERFACE at xi2 (tension #44 axis ii): C0/C1 rows must convert exactly:
     A(xi2) = Ahat/xi2, A_x(xi2) = Ahat_x/xi2 - Ahat/xi2^2, etc. A seam-residual
     check at the interface nodes runs BEFORE any Newton step.
  4. Numerics (tension #44 axis iii): evaluate E1/xi as exp(a0*xi)/s with s
     clamped; at t=1 set the exponential factors to 0 identically.
  5. Gates, in order: (a) hatted G0: the hatted residual evaluated on the adopted
     certified root (fields converted pointwise) matches the standard residual
     row-for-row on shared panels, and is small on the compactified panel;
     (b) seam residual at machine scale; (c) Newton from the adopted root;
     (d) KILL LINE: alpha within 1e-7 of -0.34471229 at (16,40)+compactified,
     eps_b = 1e-4 (battery: alpha flat to 8 decimals across XMAX 20-30, so the
     truncation-free solve must agree at that level or the formulation is wrong).

Stage-2 payoff test after the gates: eps_b ladder with NO XMAX axis (it no longer
exists), against the recorded 2D quote alpha = -0.34240 +- 4.4e-5; p-sweep windows
on the compactified panel against window wobble 0.04.

## 4. Status

Derivation VERIFIED (bundles, transport vacuity, elliptic closure).

LAYER 1 BUILT AND PASSING (polar_kernelhat.py): the compactified panel
[xi2, inf) with chain-rule Dxi/Dxi2, endpoint rows exactly zero. Self-test:
e^(a0 xi) differentiates to 1.83e-12 (deg 24, L=20), plateau structure to
machine eps. L = 20 is the working map scale; the gates will fix it properly.

NEXT (layer 2): hatted residual assembly on the compactified panel per section 1;
then gate (a) hatted-G0 against the adopted certified root (fields interpolated
onto compactified nodes for xi <= 25, two-term model beyond), then the seam rows,
then Newton. Ordering bet relevant here (#43/#33): when continuation runs in this
architecture, the first failure is predicted at eps_b/panel replication, not at
the fold detector.
