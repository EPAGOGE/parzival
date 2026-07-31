"""CANONICAL 3D-AXISYMMETRIC-EULER CORRECTIONS to the corner-regularized Boussinesq
residuals.  SELF-CONTAINED, NOT INSTALLED.  Nothing under boussinesq/ is touched.

WHAT THIS IS
------------
Hou-Luo 3D axisymmetric Euler at the wall (u1 = u_theta/R, om1 = om_theta/R,
ps1 = psi_theta/R, cylindrical radius R, wall at R = 1) maps onto the solver's corner
problem by

    Boussinesq x_phys  <->  axial z          theta <-> u1^2
    Boussinesq y_phys  <->  wall distance    Omega <-> om1 ,  Psi <-> ps1

Under that map the 3D system is the coded 2D Boussinesq system PLUS exactly four
corrections, each carrying exactly one power of the scale ratio

    tau  =  lambda  =  (blowup length) / (cylinder radius)

so that  R = 1 - tau * rho * sin(beta)  with rho = e^xi - 1 the physical corner radius.
tau = 0 is exactly 2D Boussinesq; tau = 1 is the unrescaled 3D system.

    C1  elliptic     -(d_RR + (3/R) d_R + d_zz) ps1 = om1        -> RP
    C2  wall-normal  u^R = -R d_z ps1                            -> RO, RB
    C3  axial        u^z = 2 ps1 + R d_R ps1                     -> RO, RB
    C4  stretching   D_t theta = 4 theta d_z ps1  (theta = u1^2) -> RB

SIGN CONVENTION -- fixed FROM THE CODE, not assumed
---------------------------------------------------
Rebuilding all three coded residuals from the physical PDE (adjudicate.py BLOCK 0,
three exact sympy zeros, plus a discriminator showing the flipped velocity does NOT
reproduce coded RO) forces, uniquely:

    RO = M_O * (RHS - LHS),  M_O = e^{-a0 xi} / xi            advection enters MINUS
    RB = M_B * (RHS - LHS),  M_B = e^{-(1+2 a0) xi} / xi^2    advection enters MINUS
    RP = M_P * (Lap Psi + Omega),  M_P = G1^2 e^{-a0 xi}      OPPOSITE sign
    effective 2D velocity  u = (-Psi_y, +Psi_x)  =  MINUS textbook grad-perp

Consequences carried through below: C2/C3 sit on the advection side and enter RO/RB
with the sign that makes the *whole* 3D advection equal R*(2D advection) + 2 tau Psi d_z;
C4 is an RHS source and enters RB with a PLUS; C1 follows RP's opposite convention.

VERIFICATION (scratchpad/adjudicate.py, 22/22 exact sympy zeros, 11/11 deliberately
corrupted mutants REJECTED):
    coded RO/RB/RP rebuilt from the physical PDE                    3 zeros
    handed-down C1 control regenerated at tau=1, byte-identical     1 zero
    dRO == (full 3D omega residual) - (coded RO)                    1 zero
    dRB == (full 3D theta residual) - (coded RB)                    1 zero
    dRP == (full 3D elliptic residual) - (coded RP)                 1 zero
    per-term attribution of C2/C3/C4 to their Hou-Luo mechanisms    5 zeros

WHY tau SITS INSIDE R AS WELL AS OUTSIDE (scratchpad/discriminate_tau_in_R.py):
solving  -M_O*(R-1)*Psi_x*Om_y == C2  for R returns  R = 1 - tau*rho*sin(beta).
C2/C3 therefore already carry tau inside R; freezing R = 1 - rho*sin(beta) in C1 while
C2/C3 use the tau-scaled R is two different values of the same geometric quantity.
It also parks the 3/R pole at rho ~ 1 for EVERY tau > 0, instead of pushing it out to
rho = 1/(tau sin beta).

CROSS-ROUTE STATUS
------------------
C2, C3, C4: FOUR independent derivations agree byte-for-byte -- the three workflow
routes plus the repo's own hand pass boussinesq/EULER3D_DERIVATION.md ("Route 4"),
whose recombined form reproduces dRO/dRB here to 2.0e-16 / 7.0e-16 relative on the
live grid.  SETTLED.
C1: identical in all four AT tau=1; the routes split only on the homotopy path, and
tau-inside-R wins on the internal-consistency argument above.

DOMAIN WARNING -- READ BEFORE ANY tau=1 SOLVE
---------------------------------------------
R <= 0 means the corner point has been mapped PAST the symmetry axis, outside the
cylinder the model describes.  On the shipped default grid (edges 0,2,15,25) rho
reaches 7.2e10, so tau=1 puts 85.22% of nodes at R <= 0 and the algebra will not
complain: it converges, quietly, on a region where the 3D map does not exist.
MEASURED: frac(R<=0) = 0.00% at tau=1e-11 (min R = 0.280), 49.14% at tau=1e-3,
85.22% at tau=1.  Keeping the whole default grid inside the cylinder needs
tau < 1.4e-11.

AND tau=1 IS THE WRONG ENDPOINT ANYWAY (EULER3D_DERIVATION.md section 3).  All four
corrections vanish like the blowup scale eps, which IS the Hou-Luo statement that
near-wall 3D axisymmetric Euler is asymptotically 2D Boussinesq.  So in the exact
self-similar limit alpha_3D = alpha_2D identically, and alpha(tau=1) is a finite-time
snapshot, not a self-similar exponent.  The finite, well-posed readout is
d alpha / d eps at eps = 0 -- one linear solve against the Jacobian already factored
at the certified 2D root, with these corrections as the right-hand side.  Two things
follow, and they are why this file keeps tau inside R:
  * at first order the pole never happens: 1/R -> 1 + eps*y, so the operator is
    linear and nothing divides by a sign-changing quantity;
  * no hand truncation of the handed-down control is needed.  MEASURED: the exact
    3/R self-truncates to 3 as tau -> 0 (max rel |exact - 3| = 3.0e-4 at tau=1e-14,
    3.0e-6 at tau=1e-16).  Freezing R = 1 - rho*sinb would NOT self-truncate.

OPEN, AND THE REAL FRONTIER (measured here, not raised by any of the four routes):
the tau-derivative right-hand side is dominated by the OUTER BOUNDARY, not the wall
layer.  All three components attain their sup at xi = XMAX = 25 (rho = 7.2e10), and
the sup restricted to the physically valid wall layer rho < 1 is 7 to 9 orders of
magnitude smaller:

    dRO/tau  sup 1.900e+07  |  sup over rho<1  3.021e+00  ratio 1.6e-07
    dRB/tau  sup 3.306e+05  |  sup over rho<1  8.896e-01  ratio 2.7e-06
    dRP/tau  sup 1.087e+09  |  sup over rho<1  4.137e+00  ratio 3.8e-09

So a naive d alpha / d eps solve reads out almost entirely from the far field, where
the Hou-Luo wall-layer map does not hold.  The RHS must be windowed to the wall layer
and the result quoted as a function of the window, or d alpha / d eps will simply
re-express the campaign's known (N, XMAX) two-dimensional surface with an
exponentially weighted RHS.  This is a modeling decision and is left unmade here.
"""
import numpy as np


def corrections(S, A, B, Pf, A_x, A_b, B_x, B_b, P_x, P_b, LAa, LB2b, LPp, tau):
    """3D-Euler corrections to (RO, RB, RP), gated by tau.

    Call from inside residual(), where the local bundles already exist:

        dRO, dRB, dRP = corrections(self, A, B, Pf, A_x, A_b, B_x, B_b,
                                    P_x, P_b, LAa, LB2b, LPp, self.tau)
        RO = RO + dRO ;  RB = RB + dRB ;  RP = RP + dRP

    and re-impose the special rows (rT_pin, rT_c0, rP_bedge, rP_outer, rP_c0, rP_c1,
    rP_cornerI) AFTERWARDS -- those are algebraic constraints, not PDE collocations,
    and must not receive PDE corrections.

    S supplies XI, G1, E1, cosb, sinb (a0 and mu enter only through E1/G1/the bundles).
    A_x, B_x, P_x are accepted for signature completeness; the radial derivatives
    already sit inside LAa, LB2b, LPp.

    Returns three arrays shaped like A.  At tau == 0 they are exactly zero.
    """
    if float(tau) == 0.0:                      # exact 2D, and no 0*inf can occur
        return np.zeros_like(A), np.zeros_like(A), np.zeros_like(A)

    x, G1, E1 = S.XI, S.G1, S.E1               # (Nx,1)
    cosb, sinb = S.cosb, S.sinb                # (1,Nb)

    rho = np.expm1(x)                          # physical corner radius, (Nx,1)
    R = 1.0 - tau * rho * sinb                 # cylindrical radius, (Nx,Nb)
    K = rho * E1 / G1                          # = xi*e^xi*E1 ; K(0) = 0 exactly

    # divided Cartesian-derivative bundles.  For F = xi^n f e^{c xi}:
    #   d_x(F) = (xi^n e^{c xi}/rho) (G1 cosb L_f - sinb f_b)
    #   d_y(F) = (xi^n e^{c xi}/rho) (G1 sinb L_f + cosb f_b)
    BxP = G1 * cosb * LPp - sinb * P_b         # -> Psi_x = K * BxP
    ByP = G1 * sinb * LPp + cosb * P_b         # -> M_P * Psi_y = rho * ByP
    BxA = G1 * cosb * LAa - sinb * A_b         # -> M_O * Omega_x = BxA / rho
    ByA = G1 * sinb * LAa + cosb * A_b
    BxB = G1 * cosb * LB2b - sinb * B_b        # -> M_B * theta_x = BxB / rho
    ByB = G1 * sinb * LB2b + cosb * B_b

    # excess axial velocity over 2D:  u^z - (-Psi_y) = 2 Psi + (R-1) d_R Psi
    Wz = 2.0 * Pf + sinb * ByP

    # C2 (wall-normal factor R) + C3 (axial factor R and the extra 2 ps1)
    dRO = tau * (sinb * K * BxP * ByA - K * BxA * Wz)
    dRB = tau * (sinb * K * BxP * ByB - K * BxB * Wz
                 + 4.0 * B * K * BxP)          # + C4 (theta = u1^2 stretching source)
    # C1 (the 3/R term of the axisymmetric Laplacian)
    dRP = -tau * (3.0 / R) * rho * ByP

    return dRO, dRB, dRP


# ---------------------------------------------------------------------------
# SPLIT FORM, for attribution only.  Algebraically identical to the above.
#   C2->RO = +tau*sinb*K*BxP*ByA          C2->RB = +tau*sinb*K*BxP*ByB
#   C3->RO = -tau*K*BxA*Wz                C3->RB = -tau*K*BxB*Wz
#   C4->RB = +tau*4*B*K*BxP               C1->RP = -tau*(3/R)*rho*ByP
#
# CHEAP EQUIVALENT that reuses the coded 2D bracket verbatim (also verified):
#   dRO = tau*( rho*sinb*E1*(LPp*A_b - P_b*LAa) - 2*K*Pf*BxA )
#   dRB = tau*( rho*sinb*E1*(LPp*B_b - P_b*LB2b) - 2*K*Pf*BxB + 4*B*K*BxP )
# i.e. the first piece is MINUS tau*rho*sinb times the advection term already coded.
# ---------------------------------------------------------------------------


def _selftest():
    import sys
    sys.path.insert(0, "/Users/epagogellc/parzival/boussinesq")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pcr", "/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")
    pcr = importlib.util.module_from_spec(spec)
    sys.modules["pcr"] = pcr
    spec.loader.exec_module(pcr)

    S = pcr.CornerRegSolver()
    A, B, Pf = S.A0.copy(), S.B0.copy(), S.P0.copy()
    dx = lambda F: (S.Dx @ F)
    db = lambda F: F @ S.Db.T
    x, a0, mu = S.XI, S.a0, S.mu
    A_x, A_b = dx(A), db(A)
    B_x, B_b = dx(B), db(B)
    P_x, P_b = dx(Pf), db(Pf)
    LAa = A + x * (A_x + a0 * A)
    LB2b = 2.0 * B + x * (B_x + (1.0 + 2.0 * a0) * B)
    LPp = 2.0 * Pf + x * (P_x + mu * Pf)
    args = (S, A, B, Pf, A_x, A_b, B_x, B_b, P_x, P_b, LAa, LB2b, LPp)

    print(f"solver grid: Nx={S.Nx}  Nb={S.Nb}  alpha={S.a0:.8f}  "
          f"rho_max={np.expm1(S.x[-1]):.4e}")

    ok = True
    # ---- tau = 0 : EXACTLY zero, right shape ------------------------------
    d0 = corrections(*args, tau=0.0)
    for nm, arr in zip(("dRO", "dRB", "dRP"), d0):
        shape_ok = (arr.shape == A.shape)
        zero_ok = bool(np.array_equal(arr, np.zeros_like(A))) and not np.any(np.signbit(arr))
        ok &= shape_ok and zero_ok
        print(f"  tau=0  {nm}: shape {arr.shape} == A.shape {A.shape} -> {shape_ok} ; "
              f"identically +0.0 -> {zero_ok} ; max|.| = {np.abs(arr).max():.1e}")

    # ---- tau > 0 : right shape, finite, zero on the corner circle ----------
    for tau in (1e-11, 1e-3, 1.0):
        d = corrections(*args, tau=tau)
        shp = all(a.shape == A.shape for a in d)
        fin = all(np.all(np.isfinite(a)) for a in d)
        row0 = max(float(np.abs(a[0, :]).max()) for a in d)
        Rmin = float((1.0 - tau * np.expm1(S.XI) * S.sinb).min())
        frac_bad = float(np.mean((1.0 - tau * np.expm1(S.XI) * S.sinb) <= 0.0))
        ok &= shp and fin and (row0 == 0.0)
        print(f"  tau={tau:<7.0e} shapes {shp}  finite {fin}  "
              f"max|corr| on corner row xi=0: {row0:.1e}  "
              f"min R = {Rmin:.3e}  frac(R<=0) = {frac_bad:6.2%}")

    # ---- linearity in tau (each correction carries exactly one tau, except
    #      the 1/R which is tau-dependent -> check RO/RB only) ---------------
    a1 = corrections(*args, tau=1e-6)
    a2 = corrections(*args, tau=2e-6)
    for nm, u, v in zip(("dRO", "dRB"), a1[:2], a2[:2]):
        rel = float(np.abs(2.0 * u - v).max() / max(np.abs(v).max(), 1e-300))
        ok &= rel < 1e-14
        print(f"  exact linearity in tau: {nm}  max|2*d(1e-6) - d(2e-6)|/max|d| = {rel:.2e}")

    # ---- the corrections must not blow the 2D residual up at small tau -----
    z0 = S.pack(A, B, Pf, 0.0, S.a0)
    f2d = S.residual(z0)
    n2 = S.Nx * S.Nb
    for tau in (1e-11, 1e-6, 1.0):
        d = corrections(*args, tau=tau)
        r = [float(np.abs(a).max()) for a in d]
        base = [float(np.abs(f2d[:n2]).max()), float(np.abs(f2d[n2:2*n2]).max()),
                float(np.abs(f2d[2*n2:3*n2]).max())]
        print(f"  tau={tau:<7.0e} max|dRO|={r[0]:.3e} (2D {base[0]:.3e}) "
              f"max|dRB|={r[1]:.3e} (2D {base[1]:.3e}) "
              f"max|dRP|={r[2]:.3e} (2D {base[2]:.3e})")

    print("\nSELFTEST " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
