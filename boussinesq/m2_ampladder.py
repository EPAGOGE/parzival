#!/usr/bin/env python3
"""M2 AMPLITUDE LADDER -- first measurement PAST the transient (spec:
M2_AMPLADDER_SPEC.md, staged 2026-08-02 evening).

Four marches at the PRODUCTION grid (root A, pnorm_P.npz certificate,
cholesky re-verified on load), amps 1e-6 (control) / 1e-5 / 1e-4 / 1e-3,
each: admissible perturbation (_admissible_pert, seed 0 => IDENTICAL
direction across amps, scaled -- exactly what the collapse test needs),
sign +1.0, ds=0.25, 240 steps (60 s-units).

REUSE, NOT REBUILD: _load_production_pnorm_certificate, QuasiNewtonSMarcher
(frozen LU factorized once, shared across all four amps -- it depends only
on ds), _admissible_pert, _reduced_dev, state_distance -- all imported from
march_s.py unmodified. march_s.py is NOT edited (spec: additive only).

VALIDITY / STOP RULE (spec): the TRUE nonlinear residual must converge each
step at the M1-gate tolerance (1e-10). If quasi-Newton needs >15 iterations
OR returns an unconverged residual, STOP that amp cleanly and record
step/s/V/||v||_P/raw -- the stall point is DATA (nonlinear escape), not an
error.

Outputs (new files only): M2_AMPLADDER.out (full log + analysis),
m2_ampladder_data.npz (per-step arrays per amp),
VERDICT_ORBIT_DRAFT.txt (only if some amp >= 1e-5 meets M2's criterion
with the march valid throughout -- DRAFT: milestone flips are the
operator's call).

Refusals honored: definiteness by cholesky only (the certificate loader's
own re-verify; no eigenvalue readout anywhere here); no hand-rolled
projectors; honest wall times; runs/stream_*.jsonl untouched.
"""
import pathlib
import time

import numpy as np

from march_s import (_HERE, QuasiNewtonSMarcher, _admissible_pert,
                     _load_production_pnorm_certificate, _reduced_dev,
                     state_distance)

AMPS = [1e-6, 1e-5, 1e-4, 1e-3]      # 1e-6 = control (M1's own amp)
DS = 0.25
NSTEPS = 240                          # 60 s-units
SEED = 0
STEP_TOL = 1e-10                      # M1-gate tolerance, unchanged
QN_ITER_STOP = 15                     # spec: "QN > ~15 iters => STOP"
GROWTH_TOL = 1e-12                    # M1 gate's per-step V convention
COLLAPSE_TOL = 0.10                   # 10% shape deviation = collapse broken

OUT = _HERE / "M2_AMPLADDER.out"
NPZ = _HERE / "m2_ampladder_data.npz"
DRAFT = _HERE / "VERDICT_ORBIT_DRAFT.txt"
_lines = []


def log(s=""):
    _lines.append(s)
    print(s, flush=True)


def march_one_amp(cert, M, amp):
    """One amplitude rung: perturb (fresh seed-0 rng => same direction every
    amp), march up to NSTEPS with the spec's stop rule. Returns per-step
    arrays + stop record."""
    S, z0, R0 = cert["S"], cert["z0"], cert["R0"]
    P, Z0 = cert["P"], cert["Z0"]
    n2 = S.Nx * S.Nb

    rng = np.random.default_rng(SEED)
    z = z0 + _admissible_pert(S, n2, rng, amp, z0)
    v0 = _reduced_dev(R0, Z0, z0, z)
    V0 = float(np.real(np.vdot(v0, P @ v0)))
    raw0 = state_distance(z, z0, n2)
    Vs, pnorms, raws = [V0], [float(np.sqrt(V0))], [raw0]
    resids, iters = [], []
    violations = []
    stop = None                       # None = completed all NSTEPS
    log(f"  start: |dz|_rel(raw)={raw0:.6e}  V(v)={V0:.6e}  "
        f"||v||_P={np.sqrt(V0):.6e}")

    t0 = time.time()
    for i in range(NSTEPS):
        zn, r, k = M.step(z, DS, tol=STEP_TOL, maxit=25)
        s_now = (i + 1) * DS
        if zn is None or not np.all(np.isfinite(zn)):
            stop = dict(kind="CORRECTION FAILURE", step=i + 1, s=s_now,
                        resid=float(r), iters=int(k),
                        V=Vs[-1], pnorm=pnorms[-1], raw=raws[-1])
            log(f"  STOP step {i + 1} (s={s_now:g}): frozen correction "
                f"returned no update (r={r:.3e}, it={k}) -- recorded, march "
                f"ended cleanly.")
            break
        vk = _reduced_dev(R0, Z0, z0, zn)
        Vk = float(np.real(np.vdot(vk, P @ vk)))
        pnk = float(np.sqrt(Vk)) if Vk >= 0 else float("nan")
        rawk = state_distance(zn, z0, n2)
        stalled = (r >= STEP_TOL) or (k > QN_ITER_STOP)
        rel_growth = (Vk - Vs[-1]) / max(abs(Vs[-1]), 1e-300)
        flag = "  <-- V GROWTH" if rel_growth > GROWTH_TOL else ""
        log(f"  step {i + 1:3d} (s={s_now:6.2f}): V={Vk:.6e}  "
            f"rel_dV={rel_growth:+.3e}{flag}  ||v||_P={pnk:.6e}  "
            f"raw={rawk:.6e}  r={r:.2e} it={k}")
        if stalled:
            why = (f"residual {r:.3e} >= {STEP_TOL:g}" if r >= STEP_TOL
                   else f"quasi-Newton took {k} > {QN_ITER_STOP} iterations")
            stop = dict(kind="NONLINEAR ESCAPE (stall)", step=i + 1, s=s_now,
                        resid=float(r), iters=int(k), V=Vk, pnorm=pnk,
                        raw=rawk)
            log(f"  STOP step {i + 1} (s={s_now:g}): {why} -- the march is "
                f"no longer validly converging the TRUE nonlinear residual "
                f"here. Stall point recorded as DATA (spec).")
            # the stalled step's state IS the escape record; do not march on
            Vs.append(Vk); pnorms.append(pnk); raws.append(rawk)
            resids.append(float(r)); iters.append(int(k))
            break
        if rel_growth > GROWTH_TOL:
            violations.append((i + 1, Vs[-1], Vk, rel_growth))
        Vs.append(Vk); pnorms.append(pnk); raws.append(rawk)
        resids.append(float(r)); iters.append(int(k))
        z = zn
    wall = time.time() - t0

    n_done = len(Vs) - 1
    completed = stop is None and n_done == NSTEPS
    return dict(amp=amp, Vs=np.array(Vs), pnorms=np.array(pnorms),
                raws=np.array(raws), resids=np.array(resids),
                iters=np.array(iters, dtype=int), violations=violations,
                stop=stop, n_done=n_done, completed=completed, wall_s=wall)


def s_transient_of(raws):
    """DONE.md M2 criterion: where the raw norm, AFTER its peak, first
    re-crosses BELOW its initial value. Returns (s_peak, peak_x_initial,
    s_transient or None)."""
    r0 = raws[0]
    ipk = int(np.argmax(raws))
    peak_x = raws[ipk] / max(r0, 1e-300)
    for i in range(ipk + 1, len(raws)):
        if raws[i] < r0:
            return ipk * DS, peak_x, i * DS
    return ipk * DS, peak_x, None


def main():
    t_all = time.time()
    log("=" * 78)
    log("M2 AMPLITUDE LADDER -- production grid, frozen reduced Jacobian")
    log("=" * 78)
    log(f"spec: M2_AMPLADDER_SPEC.md  amps={AMPS}  ds={DS} nsteps={NSTEPS} "
        f"(s-units={DS * NSTEPS:g})  seed={SEED} sign=+1.0  "
        f"step_tol={STEP_TOL:g}  stop: it>{QN_ITER_STOP} or residual "
        f"unconverged")
    log("")

    cert = _load_production_pnorm_certificate()
    log(f"certificate: {cert['npz_path']}  field: {cert['field_path']}")
    log(f"grid={cert['cfg']} alpha={cert['alpha']}  n_f={cert['n_f']}  "
        f"n_finite={cert['n_finite']}  N={cert['N']}")
    log(f"root ||F||_max (fresh, current solver) = "
        f"{cert['root_fmax_check']:.3e}  converged(<1e-9)={cert['root_ok']}")
    log(f"cholesky(P): RE-VERIFIED {'PASS' if cert['cholesky_ok'] else 'FAIL'}"
        f"  (stored flag={cert['stored_cholesky_ok']})")
    if not cert["root_ok"] or not cert["cholesky_ok"]:
        log("ABORT -- certificate did not re-verify; no march attempted.")
        _finish(t_all)
        return

    M = QuasiNewtonSMarcher(cert["S"], cert["R0"], cert["Lred"], cert["Z0"],
                            sign=+1.0)
    _lu, _piv, t_fact = M.frozen_lu(DS)
    log(f"frozen LU of (-sign*Lred + I/ds): {t_fact:.1f}s, factorized ONCE, "
        f"shared across all four amps (depends only on ds).")

    results = []
    for amp in AMPS:
        log("")
        log("-" * 78)
        log(f"AMP {amp:g}" + ("  (control -- M1's own amplitude)"
                               if amp == 1e-6 else ""))
        log("-" * 78)
        res = march_one_amp(cert, M, amp)
        results.append(res)
        log(f"  done: {res['n_done']}/{NSTEPS} steps in {res['wall_s']:.1f}s"
            f"  completed={res['completed']}")

    # ---------------- analysis (spec items 1-4) ----------------
    log("")
    log("=" * 78)
    log("ANALYSIS")
    log("=" * 78)

    log("")
    log("1) P-NORM MONOTONICITY PER AMP (any growth at finite amp = "
        "headline: the linear flow cannot grow in this norm)")
    for res in results:
        v = res["violations"]
        head = (f"  amp {res['amp']:g}: {len(v)} V-growth violations over "
                f"{res['n_done']} valid steps; V {res['Vs'][0]:.3e} -> "
                f"{res['Vs'][-1]:.3e}")
        if v:
            first = v[0]
            head += (f"  FIRST at step {first[0]} (s={first[0] * DS:g}), "
                     f"rel_dV={first[3]:.3e}")
        log(head)

    log("")
    log("2) TRANSIENT GEOMETRY PER AMP (DONE.md M2 criterion: raw norm "
        "re-crosses BELOW initial after its peak)")
    s_transients = {}
    for res in results:
        s_pk, pk_x, s_tr = s_transient_of(res["raws"])
        s_transients[res["amp"]] = s_tr
        tr_str = f"{s_tr:g}" if s_tr is not None else \
            "NOT REACHED within valid march"
        log(f"  amp {res['amp']:g}: s_peak={s_pk:g}  peak={pk_x:.3f}x initial"
            f"  s_transient={tr_str}")

    log("")
    log("3) ESCAPE THRESHOLD")
    for res in results:
        if res["completed"]:
            log(f"  amp {res['amp']:g}: COMPLETED all {NSTEPS} steps "
                f"(60 s-units), max QN it={int(res['iters'].max())}, "
                f"worst residual={res['resids'].max():.2e}")
        else:
            st = res["stop"]
            log(f"  amp {res['amp']:g}: {st['kind']} at step {st['step']} "
                f"(s={st['s']:g})  V={st['V']:.3e}  ||v||_P={st['pnorm']:.3e}"
                f"  raw={st['raw']:.3e}  r={st['resid']:.2e} it={st['iters']}")
    finishers = [r["amp"] for r in results if r["completed"]]
    log(f"  largest amp completing 60 s-units: "
        f"{max(finishers):g}" if finishers else
        "  NO amp completed 60 s-units")

    log("")
    log(f"4) LINEAR-REGIME COLLAPSE (shape = raw/raw0 per amp vs the 1e-6 "
        f"control; broken when |shape/shape_ctrl - 1| > {COLLAPSE_TOL:g})")
    ctrl = results[0]
    shape_ctrl = ctrl["raws"] / max(ctrl["raws"][0], 1e-300)
    for res in results[1:]:
        shape = res["raws"] / max(res["raws"][0], 1e-300)
        n = min(len(shape), len(shape_ctrl))
        dev = np.abs(shape[:n] / shape_ctrl[:n] - 1.0)
        imax = int(np.argmax(dev))
        broken = np.nonzero(dev > COLLAPSE_TOL)[0]
        where = (f"first broken at s={broken[0] * DS:g}" if broken.size
                 else f"NEVER broken over the {n - 1} common valid steps")
        log(f"  amp {res['amp']:g}: max dev {dev[imax]:.3e} at "
            f"s={imax * DS:g}; {where}")

    # ---------------- verdict draft (operator flips milestones) ------------
    qualifying = {a: s for a, s in s_transients.items()
                  if a >= 1e-5 and s is not None
                  and next(r for r in results if r["amp"] == a)["completed"]}
    if qualifying:
        dlines = ["VERDICT DRAFT (M2, amplitude ladder) -- DRAFT "
                  "DELIBERATELY: flipping done.sh milestones is the "
                  "operator's call.",
                  f"source: M2_AMPLADDER.out ({time.strftime('%Y-%m-%d')})",
                  ""]
        for a in sorted(qualifying):
            dlines.append(f"amp {a:g}: s_transient = {qualifying[a]:g} "
                          f"(march valid all {NSTEPS} steps, true residual "
                          f"converged every step)")
        DRAFT.write_text("\n".join(dlines) + "\n")
        log("")
        log(f"M2 criterion met at amp(s) {sorted(qualifying)} with valid "
            f"march throughout -- wrote {DRAFT.name} (DRAFT).")
    else:
        log("")
        log("No amp >= 1e-5 met the M2 criterion with a fully valid march "
            "-- no verdict draft written.")

    np.savez(NPZ, amps=np.array(AMPS),
             **{f"Vs_{i}": r["Vs"] for i, r in enumerate(results)},
             **{f"pnorms_{i}": r["pnorms"] for i, r in enumerate(results)},
             **{f"raws_{i}": r["raws"] for i, r in enumerate(results)},
             **{f"resids_{i}": r["resids"] for i, r in enumerate(results)},
             **{f"iters_{i}": r["iters"] for i, r in enumerate(results)},
             completed=np.array([r["completed"] for r in results]),
             n_done=np.array([r["n_done"] for r in results]),
             ds=DS, nsteps=NSTEPS, seed=SEED)
    log(f"per-step data saved: {NPZ.name}")
    _finish(t_all)


def _finish(t_all):
    log("")
    log(f"total wall: {time.time() - t_all:.1f}s")
    OUT.write_text("\n".join(_lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
