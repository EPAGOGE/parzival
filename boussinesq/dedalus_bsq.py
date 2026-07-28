#!/usr/bin/env python
"""2D inviscid Boussinesq wall-geometry singular-approach engine, in DEDALUS.

Vorticity-streamfunction form -- matches bq2 exactly (clean cross-check) and
avoids the primitive-variable pressure/tau tangle. First-class tooling: Dedalus
gives the well-conditioned sparse-tau spectral solve (subsumes Shen/solver_bank),
symbolic equations, and an adaptive IMEX timestepper.

    lap(psi) = w                              (elliptic; psi=0 at both walls)
    u = skew(grad(psi)) = (-dz psi, dx psi)   (no-penetration <=> psi=0 walls)
    dt(w) + u.grad(w) = dx(b) + nu*lap(w)     (baroclinic torque)
    dt(b) + u.grad(b) =         kappa*lap(b)
w, b advected (inviscid: no BC needed); only the Poisson ellipse takes BCs.
x-periodic (Fourier, carries the x=0 corner symmetry via mode parity), z on
[0, Lz] Chebyshev with the wall at z=0. IC: Luo-Hou-faithful buoyancy bump --
even in x (~x^2 on the axis), nonzero + concentrated at the wall; w=0.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import dedalus.public as d3


def make_filter(field, alpha=36.0, order=36, cutoff=0.65):
    """Hou-Li exponential spectral filter (J.Comput.Phys 2007) for near-
    singular flow: ~1 up to cutoff*k_max, then a very high-order rolloff that
    annihilates the aliasing-contaminated top modes while preserving the
    physical spectrum to roundoff (gate-verified 2e-16 on a smooth field)."""
    csh = field["c"].shape
    ix = np.arange(csh[0]) / max(csh[0] - 1, 1)
    iz = np.arange(csh[1]) / max(csh[1] - 1, 1)
    KX, KZ = np.meshgrid(ix, iz, indexing="ij")
    kk = np.sqrt((KX ** 2 + KZ ** 2) / 2.0)
    return np.exp(-alpha * (np.maximum(kk - cutoff, 0.0) / (1 - cutoff)) ** order)


def build_and_run(Nx, Nz, A, stop_sim_time, nu, kappa, out,
                  stepper="RK443", safety=0.2, use_filter=True,
                  run_id=None, checkpoint_wall=300.0, resume=None, ic="s2"):
    Lx, Lz = 2 * np.pi, np.pi
    coords = d3.CartesianCoordinates("x", "z")
    dist = d3.Distributor(coords, dtype=np.float64)
    xb = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx), dealias=3 / 2)
    zb = d3.ChebyshevT(coords["z"], size=Nz, bounds=(0, Lz), dealias=3 / 2)
    x = dist.local_grid(xb)
    z = dist.local_grid(zb)
    ex, ez = coords.unit_vector_fields(dist)

    psi = dist.Field(name="psi", bases=(xb, zb))
    w = dist.Field(name="w", bases=(xb, zb))
    b = dist.Field(name="b", bases=(xb, zb))
    tau1 = dist.Field(name="tau1", bases=xb)
    tau2 = dist.Field(name="tau2", bases=xb)

    lift = lambda F: d3.Lift(F, zb.derivative_basis(1), -1)
    grad_psi = d3.grad(psi) + ez * lift(tau1)      # first-order reduction
    u = d3.skew(d3.grad(psi))                       # (-dz psi, dx psi)
    dx = lambda F: d3.Differentiate(F, coords["x"])

    problem = d3.IVP([psi, w, b, tau1, tau2], namespace=locals())
    problem.add_equation("div(grad_psi) + lift(tau2) - w = 0")   # lap(psi)=w
    problem.add_equation("psi(z=0) = 0")
    problem.add_equation("psi(z=Lz) = 0")
    # advection is nonlinear (u ~ grad(psi)) -> explicit RHS; diffusion +
    # dt implicit on the LHS (IMEX). Inviscid: LHS is just dt, RHS all explicit.
    problem.add_equation("dt(w) - nu*lap(w) = dx(b) - u@grad(w)")
    problem.add_equation("dt(b) - kappa*lap(b) = -u@grad(b)")

    solver = problem.build_solver(getattr(d3, stepper))
    solver.stop_sim_time = stop_sim_time

    # --- MPI correctness (safe under `mpirun -n K`): the .max() diagnostics
    # below are LOCAL grid maxima, so reduce them across ranks; d3.integ is
    # already a global reduction. Only rank 0 touches the stream/JSON files.
    from mpi4py import MPI
    comm = dist.comm
    rank = comm.rank
    gmax = lambda v: float(comm.allreduce(float(v), op=MPI.MAX))

    # ---- checkpoint / resume (Dedalus native): dump full solver state every
    # checkpoint_wall wall-seconds so a pod death / preemption is recoverable.
    # ORDER MATTERS: load the old state BEFORE creating the handler -- an
    # 'overwrite' handler clears the dir on creation, so resume uses 'append'.
    tag = run_id or f"N{Nx}"
    ckpt_dir = pathlib.Path(out).parent / f"ckpt_{tag}"
    if resume:
        cand = sorted(ckpt_dir.glob("*.h5"))
        if isinstance(resume, str) and resume.endswith(".h5"):
            rp = resume
        elif cand:
            rp = str(cand[-1])
        else:
            raise SystemExit(f"--resume: no checkpoint in {ckpt_dir}")
        solver.load_state(rp, -1)
        print(f"  RESUMED from {rp} at t={solver.sim_time:.4f} "
              f"it={solver.iteration}", flush=True)
        ck_mode = "append"
    else:
        # --- IC FAMILY, selected by the LEADING ORDER s of the buoyancy at the corner.
        # Liu (Caltech thesis 2017) Sec 3.4/3.5 classifies the blowup by s, the leading
        # order of theta in x1 measured from the corner:
        #   s = 2  -> STABLE self-similar singularity (the Luo-Hou scenario); profiles
        #             converge; Jacobian eigenvalues have negative real parts.
        #   s >= 4 -> profiles DO NOT converge; omega -> delta-like, theta -> jump; a
        #             SECOND, SMALLER SCALE is generated. Liu: "has never been studied
        #             before", and he attacked it with FIRST-ORDER upwind + forward Euler
        #             on a 2^-18 mesh. Our spectral RK443 engine is far higher order.
        # Our corner sits at x=pi, z=0, so Liu's (x1,x2) on (-1,1)x(0,1) maps as
        #   x1 = (x-pi)/pi  =>  pi*x1 = x-pi ,   x2 = z/Lz .
        if ic == "s2":
            # b - b(pi) ~ (x-pi)^2 ; only grad b enters the vorticity equation, so the
            # additive constant is dynamically irrelevant and this IS the s=2 class.
            b["g"] = A * (0.5 * (1 - np.cos(x))) * np.exp(-30.0 * (z / Lz) ** 4)
            w["g"] = 0.0
        elif ic == "s4":
            # Liu Sec 3.5 verbatim, ported:
            #   theta = (1 - cos(pi x1))^2 (1 - x2)^2  ->  (1 + cos x)^2 (1 - z/Lz)^2
            #   omega = sin^3(pi x1)   (1 - x2)^3      ->  sin^3(x - pi) (1 - z/Lz)^3
            # theta VANISHES at the corner like (x-pi)^4  (s = 4);
            # omega is ODD about x=pi, vanishing cubically -> omega(corner) = 0.
            b["g"] = A * (1.0 + np.cos(x)) ** 2 * (1.0 - z / Lz) ** 2
            w["g"] = np.sin(x - np.pi) ** 3 * (1.0 - z / Lz) ** 3
        else:
            raise SystemExit(f"--ic must be s2 or s4, got {ic!r}")
        ck_mode = "overwrite"
    checkpoints = solver.evaluator.add_file_handler(
        str(ckpt_dir), wall_dt=checkpoint_wall, max_writes=2, mode=ck_mode)
    checkpoints.add_tasks(solver.state, layout="c")

    # Hou-Li filter: annihilates the aliasing-contaminated top ~1/3 of modes,
    # preserves the physical spectrum to roundoff (gate: 2e-16 on smooth field)
    filt = make_filter(b) if use_filter else None

    def scalar(op):
        v = op.evaluate()["g"]
        return float(v.flat[0]) if np.size(v) else 0.0
    b0sq = scalar(d3.integ(b ** 2))
    gradb = d3.grad(b)
    supgb_op = np.sqrt(gradb @ gradb)

    # tight adaptive CFL: cadence 1 (re-evaluate every step -- the front
    # sharpens fast near T*), low safety, gentle max_change so dt can only
    # creep up but drop quickly
    CFL = d3.CFL(solver, initial_dt=5e-4, cadence=1, safety=safety,
                 max_change=1.2, min_change=0.2, max_dt=1e-3, threshold=0.05)
    CFL.add_velocity(u)

    # live-stream + control files: any observer can tail the stream mid-run,
    # and drop a command in the control file to intervene WITHOUT killing it.
    stream_p = pathlib.Path(out).parent / f"stream_{tag}.jsonl"
    control_p = pathlib.Path(out).parent / f"control_{tag}.json"
    if not resume and rank == 0:
        stream_p.write_text("")             # truncate on fresh start (rank 0)
    comm.Barrier()                          # no rank appends before truncate
    ser = {"t": [], "sup_gb": [], "sup_u": [], "b2_drift": []}
    t0, brk, stop_cmd = time.time(), None, None
    while solver.proceed:
        dt = CFL.compute_timestep()
        solver.step(dt)
        if filt is not None:                 # Hou-Li filter every step
            w["c"] *= filt
            b["c"] *= filt
        if solver.iteration % 10 == 0:
            # WHERE the max sits, not just its value. Needed three separate times this
            # session: (1) at s=2 argmax|grad b| migrates interior -> wall at t~1.44, so
            # fits must start there; (2) ||omega||_inf is attained on an unrelated broad
            # structure (omega is odd at the corner), which explained a 1.703-vs-1.74 T*
            # gap; (3) at s=4 the max sits on the WALL at x/pi~0.93, AWAY from the corner
            # -- consistent with the Stage-1 boundary-point blowup of arXiv:2604.01868 --
            # so a CORNER-centred self-similar law cannot apply to it.
            _g = np.abs(supgb_op.evaluate()["g"])
            supgb = gmax(_g.max())
            _i, _j = np.unravel_index(np.argmax(_g), _g.shape) if _g.size else (0, 0)
            _xl = float(np.ravel(x)[_i]) if np.size(x) > 1 else 0.0
            _zl = float(np.ravel(z)[_j]) if np.size(z) > 1 else 0.0
            _owner = 1.0 if (_g.size and float(_g.max()) >= supgb * (1 - 1e-12)) else 0.0
            _xw = gmax(_xl * _owner); _zw = gmax(_zl * _owner)   # location on the winning rank
            supu = gmax(np.abs(u.evaluate()["g"]).max())
            drift = abs(scalar(d3.integ(b ** 2)) - b0sq) / max(abs(b0sq), 1e-300)
            ser["t"].append(solver.sim_time)
            ser["sup_gb"].append(supgb); ser["sup_u"].append(supu)
            ser["b2_drift"].append(drift)
            row = {"t": solver.sim_time, "it": int(solver.iteration),
                   "sup_gb": supgb, "sup_u": supu, "b2_drift": drift,
                   "argmax_x": _xw, "argmax_z": _zw,
                   "argmax_x_over_pi": _xw / np.pi, "argmax_z_over_Lz": _zw / Lz,
                   "dt": float(dt), "wall": round(time.time() - t0, 1)}
            if rank == 0:                     # rank 0 owns the live stream
                with open(stream_p, "a") as f:
                    f.write(json.dumps(row) + "\n")
            if solver.iteration % 200 == 0 and rank == 0:
                print(f"  t={solver.sim_time:.4f} it={solver.iteration} "
                      f"sup|grad b|={supgb:.3e} b2_drift={drift:.2e} dt={dt:.1e} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            # mid-run control: rank 0 reads + consumes, broadcasts to all ranks
            # so every rank acts collectively (a lone-rank break would deadlock).
            cmd = None
            if rank == 0 and control_p.exists():
                try:
                    cmd = json.loads(control_p.read_text())
                    control_p.unlink()       # consume once (rank 0)
                except Exception as exc:
                    print(f"  [control] bad command ignored: {exc}", flush=True)
                    cmd = None
            cmd = comm.bcast(cmd, root=0)
            if cmd:
                if cmd.get("cmd") == "checkpoint":
                    solver.evaluator.evaluate_handlers(
                        [checkpoints], wall_time=0, sim_time=solver.sim_time,
                        iteration=solver.iteration, world_time=0, timestep=dt)
                    if rank == 0:
                        print(f"  [control] checkpoint forced at "
                              f"t={solver.sim_time:.4f}", flush=True)
                elif cmd.get("cmd") == "extend":
                    solver.stop_sim_time = float(cmd["stop_time"])
                    if rank == 0:
                        print(f"  [control] stop_time -> "
                              f"{solver.stop_sim_time}", flush=True)
                elif cmd.get("cmd") == "stop":
                    stop_cmd = "control-stop"
                    break
            if drift > 3e-3:
                brk = solver.sim_time
                if rank == 0:
                    print(f"  b^2 break at t={brk:.4f} (drift {drift:.2e})", flush=True)
                break

    tt = np.array(ser["t"]); gg = np.array(ser["sup_gb"])
    dd = np.array(ser["b2_drift"]); uu = np.array(ser["sup_u"])
    trust = dd < 1e-3
    tb = float(tt[trust][-1]) if trust.any() else 0.0
    res = {"engine": "dedalus", "form": "vorticity-streamfunction",
           "meter": {"stepper": stepper, "cfl_safety": safety,
                     "hou_li_filter": bool(use_filter), "dealias": 1.5},
           "Nx": Nx, "Nz": Nz, "A": A, "nu": nu, "kappa": kappa,
           "t_trust_end": tb, "break": brk,
           "gb_ratio": float(gg[trust][-1] / gg[trust][0]) if trust.sum() > 1 else 1.0,
           "supu_trust_end": float(uu[trust][-1]) if trust.any() else 0.0,
           "wall_s": round(time.time() - t0, 1), "iters": int(solver.iteration),
           "series": {"t": tt.tolist(), "sup_gb": gg.tolist(),
                      "b2_drift": dd.tolist()}}
    if trust.sum() >= 6:
        tg, ggt = tt[trust], gg[trust]
        q = len(tg) // 3
        res["accel"] = float(np.polyfit(tg[-q:], np.log(ggt[-q:]), 1)[0]
                             / np.polyfit(tg[-2 * q:-q], np.log(ggt[-2 * q:-q]), 1)[0])
    if rank == 0:                            # one writer for the result JSON
        pathlib.Path(out).write_text(json.dumps(res, indent=2))
        print(f"[DEDALUS] {Nx}x{Nz} A={A:g}: window t<={tb:.3f} | grad-b x"
              f"{res['gb_ratio']:.1f} | accel={res.get('accel', 0):.2f} | "
              f"sup|u|_end={res['supu_trust_end']:.2f} | {res['wall_s']:.0f}s -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--Nz", type=int, default=256)
    ap.add_argument("--A", type=float, default=4.0)
    ap.add_argument("--stop", type=float, default=2.4)
    ap.add_argument("--nu", type=float, default=0.0)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--out", default="../runs/dedalus_bsq.json")
    ap.add_argument("--stepper", default="RK443",
                    help="Dedalus IMEX scheme (RK443 4th-order default)")
    ap.add_argument("--safety", type=float, default=0.2)
    ap.add_argument("--no-filter", action="store_true",
                    help="disable the Hou-Li spectral filter")
    ap.add_argument("--run-id", default=None,
                    help="tag for checkpoint/stream/control files (default N<Nx>)")
    ap.add_argument("--checkpoint-wall", type=float, default=300.0,
                    help="checkpoint every N wall-seconds (pod-death insurance)")
    ap.add_argument("--resume", nargs="?", const=True, default=None,
                    help="resume from last checkpoint (or a given .h5 path)")
    ap.add_argument("--ic", default="s2", choices=["s2", "s4"],
                    help="IC family by leading order s of the buoyancy at the corner: "
                         "s2 = Luo-Hou stable self-similar; s4 = Liu Sec 3.5 multi-scale "
                         "(profiles do NOT converge)")
    a = ap.parse_args()
    build_and_run(a.Nx, a.Nz, a.A, a.stop, a.nu, a.kappa, a.out,
                  stepper=a.stepper, safety=a.safety,
                  use_filter=not a.no_filter, run_id=a.run_id,
                  checkpoint_wall=a.checkpoint_wall, resume=a.resume, ic=a.ic)
