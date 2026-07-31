"""
Dynamic-rescaling solver for the 2D Boussinesq blowup (Luo-Hou / Chen-Hou scenario).

WHY THIS EXISTS
---------------
A direct simulation loses the singularity: the structure shrinks like L(t)~(T*-t)^gamma
while the mesh stays fixed, so you can only ever approach T* from a finite distance
(our physical runs die at t~1.53 against T*~1.705). Dynamic rescaling removes the
deadline: solve in coordinates that shrink WITH the solution, and the finite-time
blowup becomes an infinite-time approach to a STEADY STATE. Resolution stops
degrading because the object being resolved no longer shrinks.

THE TRANSFORM (Chen-Hou arXiv:2210.07191 eq 2.6-2.11)
-----------------------------------------------------
    omega(x,t) = C_w(tau)^-1 Omega(y,tau),   b(x,t) = C_b(tau)^-1 B(y,tau)
    y = x / C_l(tau),                        dt/dtau = C_w(tau)
Requiring the buoyancy forcing b_x to balance advection forces exactly one
constraint on the three scalings:      c_b = c_l + 2 c_w
leaving two free functions (c_l, c_w) -- matching the 2-parameter scaling group
    omega_{lm}(x,t) = m omega(x/l, m t),  b_{lm} = l m^2 b(x/l, m t).

The rescaled system is AUTONOMOUS:
    Omega_tau + (c_l y + U).grad Omega = c_w Omega + d_y1 B
    B_tau     + (c_l y + U).grad B     = (c_l + 2 c_w) B
    -Lap Psi = Omega,   U = grad^perp Psi = (-d_y2 Psi, d_y1 Psi),   Psi|_wall = 0
The `c_l y . grad` term is a linear OUTWARD transport and `c_w < 0` a uniform
damping; together they hold the blowing-up solution stationary in the moving frame.

NORMALIZATION (this is gauge freedom -- it is what pins the frame)
------------------------------------------------------------------
Corner form, valid when symmetry pins the singularity at y=0 on the wall:
    c_l = 2 B_{y1y1}(0) / Omega_{y1}(0)
    c_w = c_l/2 + U1_{y1}(0)
which FREEZES Omega_{y1}(0) and B_{y1y1}(0) for all tau.

GOTCHA (cost us nothing here only because it is written down):
our physical IC has omega == 0 at t=0, so Omega_{y1}(0) = 0 and c_l is singular.
The rescaled solver must be handed a LATE physical state, never t=0.

WHAT IT MEASURES
----------------
At the fixed point, gamma = -c_l/c_w is the ONLY genuine eigenvalue; the other
exponents are forced by the scaling group (||omega||~(T*-t)^-1, ||grad b||~(T*-t)^-2).
Targets: gamma = 2.9206 (Chen-Hou computer-assisted proof), 2.91 (Luo-Hou PNAS 2014).
Our physical runs already give gamma = 2.94-3.00 with T* = 1.7033 held fixed.

SYMMETRY (corner at physical x=pi, wall z=0)
    Omega odd in y1   ->  Omega(0,y2) = 0
    B     even in y1  ->  d_y1 B(0,y2) = 0
"""
import argparse, json, pathlib, time
import numpy as np
import dedalus.public as d3
from mpi4py import MPI

GAMMA_CHEN_HOU = 2.9206      # computer-assisted proof, arXiv:2210.07191 eq (2.23)
GAMMA_LUO_HOU = 2.91
SPONGE_STRENGTH = 20.0       # absorption rate in the outer ramp         # PNAS 111(36):12968 (2014)


def build(Ny1=128, Ny2=128, Ybox=8.0, dealias=3 / 2):
    """Rescaled domain: a box [0,Ybox]^2 anchored at the corner (y1=0 symmetry
    line, y2=0 wall). Chebyshev in BOTH directions -- the rescaled problem is not
    periodic, so the physical solver's RealFourier basis cannot be reused."""
    co = d3.CartesianCoordinates("y1", "y2")
    dist = d3.Distributor(co, dtype=np.float64)
    b1 = d3.ChebyshevT(co["y1"], size=Ny1, bounds=(0, Ybox), dealias=dealias)
    b2 = d3.ChebyshevT(co["y2"], size=Ny2, bounds=(0, Ybox), dealias=dealias)
    return co, dist, b1, b2


def run(Ny1, Ny2, Ybox, stop_tau, out, run_id=None, dt0=1e-3, smoke=False,
        freeze_gauge=False):
    co, dist, b1, b2 = build(Ny1, Ny2, Ybox)
    y1 = dist.local_grid(b1)
    y2 = dist.local_grid(b2)
    ey1, ey2 = co.unit_vector_fields(dist)

    Om = dist.Field(name="Om", bases=(b1, b2))
    B = dist.Field(name="B", bases=(b1, b2))
    Psi = dist.Field(name="Psi", bases=(b1, b2))
    # tau fields for the Poisson solve (2 per Chebyshev direction)
    t1 = dist.Field(name="t1", bases=b2)
    t2 = dist.Field(name="t2", bases=b2)
    t3 = dist.Field(name="t3", bases=b1)
    t4 = dist.Field(name="t4", bases=b1)

    lift1 = lambda F, n: d3.Lift(F, b1.derivative_basis(2), n)
    lift2 = lambda F, n: d3.Lift(F, b2.derivative_basis(2), n)
    d1 = lambda F: d3.Differentiate(F, co["y1"])
    d2 = lambda F: d3.Differentiate(F, co["y2"])

    # CRITICAL: take U from the CLEAN gradient. Including the tau lift terms in
    # grad(Psi) -- as the 1-wall physical solver does -- injects the tau fields
    # (pure boundary-residual numerical artifact) straight into the advecting
    # velocity. With TWO Chebyshev directions that is two contaminating lifts and
    # it is fatal: |Psi|=0.049 produced |U|=18.0 (grid-scale structure) and the run
    # died in ~5 steps. Clean gradient gives |U|=0.484 at the same instant -- a
    # factor of 37 -- and the run is then stable. Taus belong in the Poisson
    # equation ONLY.
    U = d3.skew(d3.grad(Psi))                   # (-d2 Psi, d1 Psi)

    # scalar modulation parameters, updated explicitly each step from the
    # normalization conditions (they are gauge, not dynamics)
    c_l = dist.Field(name="c_l")
    c_w = dist.Field(name="c_w")
    # Chen-Hou fixed-point values (arXiv:2210.07191 eq 2.23)
    c_l["g"] = 3.00649898
    c_w["g"] = -1.02942516

    y1f = dist.Field(name="y1f", bases=b1); y1f["g"] = y1
    y2f = dist.Field(name="y2f", bases=b2); y2f["g"] = y2

    # SPONGE. The `c_l y . grad` term is a LINEAR OUTWARD transport: it carries
    # structure to the outer edge forever. With Psi=0 imposed there, material piles
    # up against the boundary and the run dies (this -- not CFL, not aliasing -- is
    # the suspected cause of the tau~0.004 divergence). The true profile decays only
    # ALGEBRAICALLY (Omega ~ r^alpha, alpha = c_w/c_l ~ -0.34), so a hard Dirichlet
    # edge is simply the wrong far field. Absorb outgoing material in a ramp instead,
    # and keep the edge far from the core.
    Y1g, Y2g = np.meshgrid(np.ravel(y1), np.ravel(y2), indexing="ij")
    rr = np.sqrt(Y1g ** 2 + Y2g ** 2)
    r_on = 0.65 * Ybox
    ramp = np.clip((rr - r_on) / max(Ybox - r_on, 1e-12), 0.0, 1.0) ** 2
    sponge = dist.Field(name="sponge", bases=(b1, b2))
    sponge["g"] = (SPONGE_STRENGTH * ramp).reshape(sponge["g"].shape)

    problem = d3.IVP([Psi, Om, B, t1, t2, t3, t4], namespace=locals())
    problem.add_equation("lap(Psi) + lift1(t1,-1) + lift1(t2,-2)"
                         " + lift2(t3,-1) + lift2(t4,-2) = -Om")
    problem.add_equation("Psi(y2=0) = 0")               # wall: no through-flow
    problem.add_equation("Psi(y2=Ybox) = 0")            # far field
    problem.add_equation("Psi(y1=0) = 0")               # symmetry line
    problem.add_equation("Psi(y1=Ybox) = 0")
    # rescaled evolution: everything nonlinear/variable-coefficient on the RHS
    problem.add_equation(
        "dt(Om) + sponge*Om = c_w*Om + d1(B) - (c_l*y1f + U@ey1)*d1(Om) - (c_l*y2f + U@ey2)*d2(Om)")
    problem.add_equation(
        "dt(B) + sponge*B = (c_l + 2*c_w)*B - (c_l*y1f + U@ey1)*d1(B) - (c_l*y2f + U@ey2)*d2(B)")

    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = stop_tau

    # --- initial profile: correct SYMMETRY and far-field decay, amplitude O(1).
    # The self-similar profile is an attracting fixed point, so a generic seed of
    # the right symmetry class relaxes onto it; we are not trying to guess it.
    # The SIGN relation between Om and B is not free: the corner formula
    # c_l = 2 B_{y1y1}(0)/Om_{y1}(0) must come out POSITIVE (outward transport).
    # A naive seed (Om ~ +y1 e^-r2, B ~ +e^-r2) gives c_l = -32 and c_w > 0, i.e.
    # growth instead of damping -> the rescaled frame blows up immediately.
    # So we scale the Om amplitude to put c_l exactly on the Chen-Hou value.
    # SEED SIGN: the OVERALL sign of (Om,B) leaves c_l invariant (it appears in
    # both numerator and denominator of 2 B_y1y1(0)/Om_y1(0)) but FLIPS the induced
    # velocity, hence c_w. Measured at tau~0.12, N=64:
    #   seed +:  U1_y1(0) = +2.396 -> c_w = +2.838 -> gamma = -0.312   (WRONG SIGN)
    #   seed -:  U1_y1(0) = -3.148 -> c_w = -2.287 -> gamma = +0.753   (all signs OK)
    # Chen-Hou target: U1_y1(0) = -2.5327, c_w = -1.0294, gamma = +2.9206.
    SEED_SIGN = -1.0
    A_GAUSS, CL_SEED = 8.0, 3.0
    Y1, Y2 = np.meshgrid(np.ravel(y1), np.ravel(y2), indexing="ij")
    r2 = Y1 ** 2 + Y2 ** 2
    B["g"] = (SEED_SIGN * np.exp(-A_GAUSS * r2)).reshape(B["g"].shape)   # EVEN in y1
    # B_{y1y1}(0) = -2a  =>  Om_{y1}(0) must equal 2*(-2a)/CL_SEED
    amp = 2.0 * (-2.0 * A_GAUSS) / CL_SEED
    Om["g"] = (SEED_SIGN * amp * Y1 * np.exp(-A_GAUSS * r2)).reshape(Om["g"].shape)  # ODD

    tag = run_id or f"R{Ny1}"
    outp = pathlib.Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    stream = outp.parent / f"rstream_{tag}.jsonl"
    rank = dist.comm.rank
    if rank == 0:
        stream.write_text("")

    # CRITICAL: ChebyshevT collocation is Chebyshev-GAUSS (roots) and therefore
    # EXCLUDES the endpoints, so `field['g'].flat[0]` is the first INTERIOR point,
    # not the corner. Reading the gauge there makes c_l/c_w drift to the wrong
    # sign and the run destroys itself (this was not a CFL problem -- cutting dt
    # 100x changed nothing). Must interpolate onto the boundary explicitly.
    corner_ops = {
        "Om_y1": d1(Om)(y1=0)(y2=0),
        "B_y11": d1(d1(B))(y1=0)(y2=0),
        "U1_y1": d1(U @ ey1)(y1=0)(y2=0),
    }

    def corner_normalization():
        """c_l = 2 B_{y1y1}(0)/Om_{y1}(0);  c_w = c_l/2 + U1_{y1}(0),
        evaluated by spectral interpolation AT the wall/symmetry corner (0,0)."""
        try:
            vals = {}
            for k, op in corner_ops.items():
                v = op.evaluate()["g"]
                vals[k] = float(np.ravel(v)[0]) if np.size(v) else 0.0
            a = vals["Om_y1"]
            if abs(a) < 1e-30:
                return None, None            # the omega==0 singular case
            cl = 2 * vals["B_y11"] / a
            return cl, 0.5 * cl + vals["U1_y1"]
        except Exception:
            return None, None

    # Hou-Li style exponential filter on both Chebyshev directions. The physical
    # solver needed this to survive N=768; the rescaled advection (c_l y + U).grad
    # is at least as aliasing-prone, and without it this run dies at tau~0.004.
    def make_filter(field):
        sh = field["c"].shape
        ax = []
        for n in sh:
            k = np.arange(n) / max(n - 1, 1)
            ax.append(np.exp(-36.0 * k ** 36))
        f = ax[0][:, None] * ax[1][None, :] if len(sh) == 2 else ax[0]
        return f.reshape(sh)

    USE_FILTER = False
    filt = make_filter(Om)

    ser, t0 = [], time.time()
    while solver.proceed:
        cl, cw = (None, None) if freeze_gauge else corner_normalization()
        if cl is not None and np.isfinite(cl) and np.isfinite(cw):
            # relax toward the normalization to keep the gauge update stable
            c_l["g"] = 0.9 * float(c_l["g"].flat[0]) + 0.1 * np.clip(cl, -50, 50)
            c_w["g"] = 0.9 * float(c_w["g"].flat[0]) + 0.1 * np.clip(cw, -50, 50)
        solver.step(dt0)
        if USE_FILTER:            # OFF: tested and it made things WORSE (gamma
            Om["c"] *= filt       # 2.85 -> 0.10) without moving the divergence
            B["c"] *= filt        # point, because damping Chebyshev coefficients
        if solver.iteration % 20 == 0:   # corrupts the boundary interpolation

            cl_v = float(c_l["g"].flat[0]); cw_v = float(c_w["g"].flat[0])
            gam = -cl_v / cw_v if abs(cw_v) > 1e-12 else float("nan")
            om_max = float(dist.comm.allreduce(float(np.abs(Om["g"]).max()), op=MPI.MAX))
            row = dict(tau=solver.sim_time, it=int(solver.iteration),
                       c_l=cl_v, c_w=cw_v, gamma=gam, om_max=om_max,
                       wall=round(time.time() - t0, 1))
            ser.append(row)
            if rank == 0:
                with open(stream, "a") as f:
                    f.write(json.dumps(row) + "\n")
                if solver.iteration % 200 == 0:
                    print(f"  tau={solver.sim_time:.4f} c_l={cl_v:+.4f} c_w={cw_v:+.4f} "
                          f"gamma={gam:.4f} |Om|={om_max:.3e}", flush=True)
            if not np.isfinite(om_max) or om_max > 1e12:
                if rank == 0:
                    print(f"  DIVERGED at tau={solver.sim_time:.4f}", flush=True)
                break

    res = dict(engine="rescale", Ny1=Ny1, Ny2=Ny2, Ybox=Ybox,
               gamma_chen_hou=GAMMA_CHEN_HOU, gamma_luo_hou=GAMMA_LUO_HOU,
               gamma_final=ser[-1]["gamma"] if ser else None,
               c_l_final=ser[-1]["c_l"] if ser else None,
               c_w_final=ser[-1]["c_w"] if ser else None,
               iters=int(solver.iteration), wall_s=round(time.time() - t0, 1),
               series=ser[-400:])
    if rank == 0:
        outp.write_text(json.dumps(res, indent=2))
        g = res["gamma_final"]
        print(f"[RESCALE] {Ny1}x{Ny2} gamma={g if g is None else round(g,4)} "
              f"(target {GAMMA_CHEN_HOU}) c_l={res['c_l_final']} c_w={res['c_w_final']} "
              f"-> {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ny1", type=int, default=128)
    ap.add_argument("--Ny2", type=int, default=128)
    ap.add_argument("--Ybox", type=float, default=8.0)
    ap.add_argument("--stop", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", default="../runs/rescale.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--freeze-gauge", action="store_true",
                    help="hold c_l,c_w at the Chen-Hou fixed point (isolates the gauge loop)")
    a = ap.parse_args()
    run(a.Ny1, a.Ny2, a.Ybox, a.stop, a.out, run_id=a.run_id, dt0=a.dt, smoke=a.smoke,
        freeze_gauge=a.freeze_gauge)
