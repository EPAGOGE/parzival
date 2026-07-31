"""
Newton solve for the self-similar BLOWUP PROFILE of 2D Boussinesq at a wall.

WHY NEWTON AND NOT TIME-MARCHING
--------------------------------
Time-marching the rescaled equations (`rescale.py`) is stable and well-conditioned
after the fixes, and it still does not find the profile: c_l -> 0, i.e. it relaxes to
a NON-concentrating state. The blowup profile is a fixed point that time-marching only
reaches from inside a narrow basin, and two independent analytic seeds with the
correct symmetry gave U1_y1(0) of OPPOSITE SIGN (-3.148 vs +0.73, target -2.5327). The
basin cannot be guessed into.

A fixed point is a ROOT-FIND. Newton converges from a far wider set, needs no basin,
and -- the real prize -- its Jacobian spectrum IS the linear stability of the blowup.
A profile without a stability statement is not a result.

THE SYSTEM (steady state of the rescaled equations; Chen-Hou arXiv:2210.07191)
------------------------------------------------------------------------------
    0 = c_w*Om + d1(B) - (c_l*y + U).grad Om
    0 = (c_l + 2*c_w)*B - (c_l*y + U).grad B
    -Lap Psi = Om,   U = skew(grad Psi),   Psi|_wall = 0
Unknowns: Om(y), B(y), Psi(y) AND the two scalars c_l, c_w.
Counting: the two extra scalar unknowns need two extra scalar equations -- these are
the GAUGE normalizations. Fixing c_l,c_w instead and dropping them makes the system
over-determined, whose only generic solution is Om=B=0 (the trivial one). So the
scalars MUST be solved for, not imposed.

GAUGE (weighted, for conditioning -- this matters)
-------------------------------------------------
Point-value normalizations (`c_l = 2 B_y1y1(0)/Om_y1(0)`) are a ratio of a second to
a first derivative evaluated AT a corner: measured condition number ~7.8e2 with
unweighted integrals and the control parameter oscillated -33/-116/+178/+27.
Weighted integrals with `wt = (1+r^2)^-2` gave cond ~1.2e1 -- 60x better -- because
with `Om ~ r^alpha` (alpha = c_w/c_l ~ -0.342) the integrand of the UNWEIGHTED
`int Om^2` goes like r^(2alpha+1) = r^(+0.315): it GROWS, so an unweighted norm gauges
the OUTER BOUNDARY rather than the singularity. The weight localizes on the core.

    int wt*Om^2      = E1     (pins amplitude)
    int wt*r^2*Om^2  = E2     (pins length scale)

TARGETS (verified citations, for scoring the answer -- NOT fed to the solver)
    c_l = 3.00649898, c_w = -1.02942516, u_x(0) = -2.532674,
    gamma = -c_l/c_w = 2.9205600                 [Chen-Hou arXiv:2210.07191 eq 2.23]
    gamma = 2.91                                 [Luo-Hou PNAS 111(36):12968 (2014)]
    far field: Om ~ r^alpha, alpha = c_w/c_l ~ -0.3424

SYMMETRY (confirmed in our own data to 12 digits: |omega|(x=pi) = 2.2e-14)
    Om ODD in y1  -> Om(0,y2) = 0 ;  B EVEN in y1 -> d1(B)(0,y2) = 0
"""
import argparse, json, pathlib
import numpy as np
import dedalus.public as d3

C_L_TARGET, C_W_TARGET = 3.00649898, -1.02942516
GAMMA_TARGET = 2.9205600
UX0_TARGET = -2.532674
ALPHA = C_W_TARGET / C_L_TARGET          # -0.34240..., the far-field exponent


def build(N=48, Ybox=8.0, dealias=3 / 2):
    co = d3.CartesianCoordinates("y1", "y2")
    dist = d3.Distributor(co, dtype=np.float64)
    b1 = d3.ChebyshevT(co["y1"], size=N, bounds=(0, Ybox), dealias=dealias)
    b2 = d3.ChebyshevT(co["y2"], size=N, bounds=(0, Ybox), dealias=dealias)
    return co, dist, b1, b2


def run(N=48, Ybox=8.0, iters=25, tol=1e-8, seed_sign=-1.0, out=None, verbose=True,
        damping=1.0, pin_c_l=False):
    co, dist, b1, b2 = build(N, Ybox)
    y1 = dist.local_grid(b1); y2 = dist.local_grid(b2)
    ey1, ey2 = co.unit_vector_fields(dist)

    Om = dist.Field(name="Om", bases=(b1, b2))
    B = dist.Field(name="B", bases=(b1, b2))
    Psi = dist.Field(name="Psi", bases=(b1, b2))
    c_l = dist.Field(name="c_l")             # scalar unknowns
    c_w = dist.Field(name="c_w")
    t1 = dist.Field(name="t1", bases=b2); t2 = dist.Field(name="t2", bases=b2)
    t3 = dist.Field(name="t3", bases=b1); t4 = dist.Field(name="t4", bases=b1)
    # tau fields for the two transported equations (first-order in each direction)
    # ONE tau per transport equation, in y1 ONLY. Characteristic analysis:
    #   y1=0: U1=-d2(Psi)=0 (Psi=0 there) and c_l*y1=0  -> CHARACTERISTIC (zero speed)
    #   y2=0: U2= d1(Psi)=0 (Psi=0 on wall)             -> CHARACTERISTIC
    #   outer edges: c_l*y > 0                          -> OUTFLOW
    # There is NO inflow boundary: the profile is selected by regularity/decay, not by
    # boundary data. Imposing Om=B=0 at the OUTFLOW edge over-determined the system and
    # made the Jacobian singular (raw step norm pinned at ~1e12 for 14 iterations while
    # the residual GREW 30x above the seed). Only the two y1=0 symmetry conditions are
    # legitimate, so only two taus.
    s1 = dist.Field(name="s1", bases=b2)
    q1 = dist.Field(name="q1", bases=b2)

    lift1 = lambda F, n: d3.Lift(F, b1.derivative_basis(2), n)
    lift2 = lambda F, n: d3.Lift(F, b2.derivative_basis(2), n)
    L1 = lambda F, n: d3.Lift(F, b1.derivative_basis(1), n)
    L2 = lambda F, n: d3.Lift(F, b2.derivative_basis(1), n)
    d1 = lambda F: d3.Differentiate(F, co["y1"])
    d2 = lambda F: d3.Differentiate(F, co["y2"])
    # CLEAN gradient: never put tau lifts into the advecting velocity (that bug gave
    # |Psi|=0.049 -> |U|=18.0 grid-scale garbage in the time-marched version)
    U = d3.skew(d3.grad(Psi))

    y1f = dist.Field(bases=b1); y1f["g"] = y1
    y2f = dist.Field(bases=b2); y2f["g"] = y2
    Y1, Y2 = np.meshgrid(np.ravel(y1), np.ravel(y2), indexing="ij")
    R2 = Y1 ** 2 + Y2 ** 2
    r2f = dist.Field(bases=(b1, b2)); r2f["g"] = R2.reshape(r2f["g"].shape)
    wt = dist.Field(bases=(b1, b2)); wt["g"] = ((1.0 + R2) ** -2.0).reshape(wt["g"].shape)

    # ---- seed with the correct far field: Om ~ y1*(1+r^2)^((alpha-1)/2) ~ r^alpha
    # B MUST VANISH AT THE CORNER. At (0,0) both U components vanish (Psi=0 along both
    # lines through the origin) and c_l*y=0, so the B equation collapses to
    #     (c_l + 2 c_w) * B(0,0) = 0,  and  c_l + 2 c_w = 0.947 != 0  =>  B(0,0) = 0.
    # A seed with B(0,0) = const violates this by ~44% AT THE CORNER (measured: worst
    # |R_B| at y1=y2=0.005), which is what drove c_w positive and blew Newton up.
    # B even in y1 + B(0)=0 => B ~ y1^2 near the corner, i.e. Chen-Hou freeze
    # B_y1y1(0), not B(0). Far field still B ~ r^(1+2alpha).
    p = (1 + 2 * ALPHA) / 2.0
    A0 = 1.0
    C0 = C_L_TARGET * A0 / 4.0          # B_y1y1(0) = 2*C0 => c_l = 2*(2C0)/A0 = 4C0/A0
    B["g"] = (seed_sign * C0 * Y1 ** 2 * (1 + R2) ** (p - 1)).reshape(B["g"].shape)
    Om["g"] = (seed_sign * A0 * Y1 * (1 + R2) ** ((ALPHA - 1) / 2)).reshape(Om["g"].shape)
    c_l["g"] = C_L_TARGET
    c_w["g"] = C_W_TARGET

    integ = lambda ex: d3.Integrate(d3.Integrate(ex, co["y1"]), co["y2"])
    val = lambda ex: float(np.ravel(ex.evaluate()["g"])[0]) if np.size(ex.evaluate()["g"]) else 0.0
    # gauge targets are taken FROM THE SEED, so the seed satisfies them exactly at
    # init -- Newton then moves the shape, not the normalization
    E1 = val(integ(wt * Om ** 2))
    E2 = val(integ(wt * r2f * Om ** 2))
    if verbose:
        print(f"  gauge targets from seed:  E1={E1:.6e}  E2={E2:.6e}")

    hom = y1f * d1(Psi) + y2f * d2(Psi) - (2 + c_w / c_l) * Psi
    CLT = C_L_TARGET; CWT = C_W_TARGET   # module globals are NOT in namespace=locals(); equation
                           # strings are eval'd against that namespace, so the pinned
                           # value must be a LOCAL defined before the capture below.
    problem = d3.NLBVP([Psi, Om, B, c_l, c_w, t1, t2, t3, t4, s1, q1],
                       namespace=locals())
    problem.add_equation("lap(Psi) + lift1(t1,-1) + lift1(t2,-2)"
                         " + lift2(t3,-1) + lift2(t4,-2) + Om = 0")
    # FAR FIELD. `Psi = 0` at the outer edges is INCOMPATIBLE with the profile:
    # Psi ~ r^(2+alpha) with 2+alpha = 1.658, i.e. it GROWS. Forcing zero there means
    # the true profile is not a solution of the discrete system at all, which is why
    # every gauge/pin permutation collapsed to a degenerate root.
    # FIX: impose the exact HOMOGENEITY relation instead. For Psi ~ r^d f(theta),
    # Euler's identity gives y.grad Psi = d*Psi -- true for ANY angular structure, and
    # homogeneous, so it introduces no unknown constant and no scale. d = 2 + c_w/c_l.
    problem.add_equation("Psi(y1=0) = 0")            # symmetry line
    problem.add_equation("Psi(y2=0) = 0")            # wall: no through-flow
    problem.add_equation("hom(y1=Ybox) = 0")         # algebraic far field, not zero
    problem.add_equation("hom(y2=Ybox) = 0")
    problem.add_equation("c_w*Om + d1(B) - (c_l*y1f + U@ey1)*d1(Om)"
                         " - (c_l*y2f + U@ey2)*d2(Om) + L1(s1,-1) = 0")
    problem.add_equation("(c_l + 2*c_w)*B - (c_l*y1f + U@ey1)*d1(B)"
                         " - (c_l*y2f + U@ey2)*d2(B) + L1(q1,-1) = 0")
    problem.add_equation("Om(y1=0) = 0")                 # Om ODD about the corner
    problem.add_equation("d1(B)(y1=0) = 0")              # B EVEN about the corner
    if pin_c_l == "both":
        # BRANCH SELECTION, correct version. I earlier claimed fixing BOTH scalars
        # leaves only the trivial solution -- that was WRONG: the system is not
        # scale-invariant (U.grad Om is quadratic, everything else linear), so the
        # AMPLITUDE is determined and a nonzero solution generically exists. So fix
        # both scalars and drop BOTH integral gauges; the count stays balanced
        # (2 extra unknowns, 2 pin equations). If this converges to a nonzero field
        # with small residual, that IS the profile at the published exponents --
        # a validation of SHAPE, not an independent derivation of the exponents.
        problem.add_equation("c_l = CLT")
        problem.add_equation("c_w = CWT")
    else:
        problem.add_equation("integ(wt*Om**2) = E1")     # gauge 1: amplitude
    if pin_c_l is True:
        # BRANCH SELECTION (continuation step 1). Newton with both integral gauges
        # converges to the c_l -> 0 NON-CONCENTRATING root (c_l drifted
        # 3.0065->0.11 while the residual fell), which solves the equations but is
        # not the blowup profile. Pinning c_l selects the right branch; the length-
        # scale gauge is dropped to keep the count balanced. Step 2 is to RELEASE
        # c_l and re-converge so the final value is independently determined --
        # until that is done, c_l here is an INPUT, not a measurement.
        problem.add_equation("c_l = CLT")
    elif not pin_c_l:
        problem.add_equation("integ(wt*r2f*Om**2) = E2")   # gauge 2: length scale

    RES_OM = c_w*Om + d1(B) - (c_l*y1f + U@ey1)*d1(Om) - (c_l*y2f + U@ey2)*d2(Om)
    RES_B  = (c_l + 2*c_w)*B - (c_l*y1f + U@ey1)*d1(B) - (c_l*y2f + U@ey2)*d2(B)
    def resid():
        a = np.abs(RES_OM.evaluate()["g"]); b_ = np.abs(RES_B.evaluate()["g"])
        return float(a.max()), float(b_.max())
    solver = problem.build_solver()
    _r0 = resid()
    if verbose:
        print(f"  SEED residual: |R_Om|={_r0[0]:.4e}  |R_B|={_r0[1]:.4e}   <- verify at init")
    hist = []
    if verbose:
        print(f"  {'iter':>4} {'|pert|':>12} {'c_l':>11} {'c_w':>11} {'gamma':>10}")
        _cl0 = float(np.ravel(c_l['g'])[0]); _cw0 = float(np.ravel(c_w['g'])[0])
        print(f"  {'seed':>4} {'-':>12} {_cl0:11.6f} {_cw0:11.6f} {-_cl0/_cw0:10.5f}")
    pert = np.inf
    for it in range(1, iters + 1):
        # DAMPED Newton: the seed carries a ~150% relative residual (measured), so a
        # full step overshoots catastrophically (|pert| ~ 4e12 at damping=1).
        solver.newton_iteration(damping=damping)
        pert = float(np.sqrt(sum(float(np.sum(np.asarray(p_["c"]) ** 2))
                                 for p_ in solver.perturbations)))
        cl = float(np.ravel(c_l["g"])[0]); cw = float(np.ravel(c_w["g"])[0])
        g = -cl / cw if abs(cw) > 1e-300 else float("nan")
        hist.append(dict(it=it, pert=float(pert), c_l=cl, c_w=cw, gamma=g))
        nOm = float(np.abs(Om["g"]).max()); nB = float(np.abs(B["g"]).max())
        hist[-1]["amp_Om"] = nOm; hist[-1]["amp_B"] = nB
        if verbose:
            print(f"  {it:>4} {pert:12.4e} {cl:11.6f} {cw:11.6f} {g:10.5f}"
                  f"  ||Om||={nOm:.4e} ||B||={nB:.4e}", flush=True)
        ro, rb = resid()
        hist[-1]["res_Om"] = ro; hist[-1]["res_B"] = rb
        if verbose:
            print(f"       residual: |R_Om|={ro:.4e}  |R_B|={rb:.4e}", flush=True)
        if not all(np.isfinite(x) for x in (ro, rb, cl, cw)):
            if verbose: print("  -> Newton diverged (non-finite)"); break
        if max(ro, rb) < tol:
            if verbose: print("  -> converged"); break
    res = dict(N=N, Ybox=Ybox, history=hist,
               c_l_target=C_L_TARGET, c_w_target=C_W_TARGET, gamma_target=GAMMA_TARGET,
               c_l=hist[-1]["c_l"] if hist else None,
               c_w=hist[-1]["c_w"] if hist else None,
               gamma=hist[-1]["gamma"] if hist else None,
               pert_final=hist[-1]["pert"] if hist else None)
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--Ybox", type=float, default=8.0)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--seed-sign", type=float, default=-1.0)
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument("--pin", default="none", choices=["none","c_l","both"])
    ap.add_argument("--out", default="../runs/profile_newton.json")
    a = ap.parse_args()
    r = run(N=a.N, Ybox=a.Ybox, iters=a.iters, seed_sign=a.seed_sign, out=a.out,
            damping=a.damping,
            pin_c_l=("both" if a.pin=="both" else (True if a.pin=="c_l" else False)))
    print(f"\n[NEWTON] c_l={r['c_l']} (target {C_L_TARGET})")
    print(f"         c_w={r['c_w']} (target {C_W_TARGET})")
    print(f"         gamma={r['gamma']} (target {GAMMA_TARGET})")
