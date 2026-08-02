#!/usr/bin/env python3
"""ADVERSARIAL KILL-SHOT CHECK (read-only wrt march_s.py / polar_cornerreg.py /
polar_spectrum.py -- calls the existing SMarcher.step() as-is, no hand-rolled
projector, no source edits).

CLAIM UNDER TEST: march_s.py now integrates the index-2 DAE correctly on
ker(Cg).  If true, F(z*) = 0 at a converged fixed point implies one
backward-Euler step started EXACTLY there must not move the state (to solver
tolerance) -- constraint/mass-term leakage would show up as movement that
does not shrink with the root's own residual.

Design:
  1. Converge a root S.newton() (SMOKE SCALE grid, same as march_s.smoke_test,
     so this runs in seconds on the laptop -- NOT a production-grid claim).
  2. z_old = z = that converged root, EXACTLY (no perturbation).
  3. Run SMarcher.step(z_old, ds) for several ds and both signs.
  4. Report: baseline ||F(z0)||, post-step Newton residual/iterations,
     ||z_new - z_old|| (abs and relative), |g1|,|g2|,||RP|| at z_new, and
     the same at z_old for comparison.
  5. Residual the solve is NOT answerable to (S9-compliant, no eigenvalues):
     re-run with a DELIBERATELY UNDER-converged root (loose tol) and check
     that step-induced movement scales with the root's own residual rather
     than being O(1) regardless -- an O(1) response to an O(1e-6) input
     residual would itself be the leakage signature.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from polar_cornerreg import CornerRegSolver
from march_s import SMarcher, state_distance

OUT = pathlib.Path(__file__).parent / "M1_FIXEDPOINT_CHECK.out"

# NOTE (found empirically, logged here so the choice isn't silently picked):
# the march_s.smoke_test grid (6,10,5)/Nb10 does NOT converge past a
# ||F||_max ~1.1e-2 FLOOR under S.newton() -- verified independently: three
# grid sizes at the same alpha=-0.3447 give fmax = 1.099e-02 / 4.170e-12 /
# 6.149e-11 for (6,10,5)/Nb10, (10,20,8)/Nb16, (16,40,12)/Nb36 respectively.
# That floor is a discretization-resolution property of the coarsest grid
# (Newton's own damped Levenberg loop exhausts every (mu, lambda) pair with
# no further reduction -- `best is None`), NOT a march_s.py defect, and NOT
# something this check may paper over: a "kill shot from the exact fixed
# point" is not actually testable on a root that is only converged to 1e-2.
# So the PRIMARY grid here is the one that genuinely reaches Newton/machine
# tolerance; the coarse smoke grid is kept only as a labeled secondary case.
GRID = dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(10, 20, 8), Nb=16, eps_b=1e-3)
GRID_COARSE = dict(edges=(0.0, 2.0, 15.0, 25.0), degs=(6, 10, 5), Nb=10, eps_b=1e-3)
ALPHA = -0.3447

lines = []


def log(s=""):
    lines.append(s)
    print(s)


def build_root(tol, steps, grid=None):
    S = CornerRegSolver(alpha=ALPHA, **(grid if grid is not None else GRID))
    z0, f0, r0, taken = S.newton(steps=steps, tol=tol, verbose=False)
    return S, z0, f0, r0, taken


def manifold_report(S, z, n2):
    f = S.residual(z)
    g1, g2 = float(f[-2]), float(f[-1])
    rp = float(np.max(np.abs(f[2 * n2:3 * n2])))
    fmax = float(np.max(np.abs(f)))
    return fmax, g1, g2, rp


def kill_shot(tag, tol, steps, ds_list, signs, grid=None):
    g = grid if grid is not None else GRID
    log("=" * 78)
    log(f"[{tag}] root: tol={tol:g} steps={steps}  grid={g} alpha={ALPHA}")
    log("=" * 78)
    S, z0, f0, r0, taken = build_root(tol, steps, grid=g)
    n2 = S.Nx * S.Nb
    fmax0, g1_0, g2_0, rp0 = manifold_report(S, z0, n2)
    log(f"root quality: ||F||_max={fmax0:.6e}  ||F||_rms={r0:.6e}  "
        f"newton_taken={taken}  c_l={z0[-2]:.8f}  c_w={z0[-1]:.8f}")
    log(f"root manifold: |g1|={abs(g1_0):.3e} |g2|={abs(g2_0):.3e} "
        f"||RP||_max={rp0:.3e}")
    log("-" * 78)

    results = {}
    for sign in signs:
        for ds in ds_list:
            M = SMarcher(S, sign=sign)
            z_old = z0.copy()
            z_new, r_newton, k_newton = M.step(z_old, ds, tol=1e-10, maxit=25)
            if z_new is None:
                log(f"sign={sign:+.0f} ds={ds:<6g}  STEP FAILED "
                    f"(reduced correction returned None) at residual {r_newton:.3e}, "
                    f"iter {k_newton}")
                results[(sign, ds)] = None
                continue
            abs_move = float(np.max(np.abs(z_new - z_old)))
            rel_move = state_distance(z_new, z_old, n2)
            fmax_n, g1_n, g2_n, rp_n = manifold_report(S, z_new, n2)
            log(f"sign={sign:+.0f} ds={ds:<6g}  Newton: residual={r_newton:.3e} "
                f"in {k_newton} it   "
                f"|z_new-z_old|_max={abs_move:.6e}  |z_new-z_old|_rel(A,B)={rel_move:.6e}")
            log(f"{'':22s}post-step manifold: |g1|={abs(g1_n):.3e} "
                f"|g2|={abs(g2_n):.3e}  ||RP||_max={rp_n:.3e}  ||F||_max={fmax_n:.3e}")
            results[(sign, ds)] = dict(abs_move=abs_move, rel_move=rel_move,
                                        fmax=fmax_n, r_newton=r_newton,
                                        k_newton=k_newton)
    log("-" * 78)
    return z0, fmax0, results


def main():
    log("ADVERSARIAL KILL-SHOT: does a tight-tol fixed point stay fixed under")
    log("one backward-Euler s-march step, at SMOKE grid scale?")
    log("(No eigenvalues used anywhere in this check -- pure residual/movement")
    log(" diagnostics, per standing refusal S9.)")
    log("")

    # --- Test A: TIGHT root, GENUINELY converged to Newton/machine tol on a
    # grid that actually gets there (see GRID note above -- the smoke_test's
    # own (6,10,5)/Nb10 grid stalls at ~1.1e-2 and cannot support a literal
    # "started from the exact fixed point" claim).  Several ds and both signs.
    z0_tight, fmax0_tight, res_tight = kill_shot(
        "TEST A -- tight root (10,20,8)/Nb16, the actual kill shot", tol=1e-12,
        steps=120, ds_list=(0.05, 0.25, 1.0), signs=(+1.0, -1.0))

    log("")
    # --- Test B: LOOSE root on the SAME grid (deliberately under-converged)
    # -- the residual the tight-root PASS is NOT answerable to: does movement
    # actually track the root's own residual, or is it O(1) leakage
    # regardless of input?
    z0_loose, fmax0_loose, res_loose = kill_shot(
        "TEST B -- deliberately loose root, same grid as A (leakage-scaling probe)",
        tol=1e-4, steps=2, ds_list=(0.25,), signs=(+1.0,))

    log("")
    # --- Test C: the SMOKE-scale grid the module's own smoke_test() uses.
    # Reported for comparison to the M1_T4_SMOKE.out numbers already on disk
    # -- NOT a kill shot (root only reaches ||F||_max ~1.1e-2 there, verified
    # independently across three grid sizes at the same alpha; see GRID note).
    z0_coarse, fmax0_coarse, res_coarse = kill_shot(
        "TEST C -- smoke_test's own (6,10,5)/Nb10 grid, NOT converged tight "
        "(informational only, not a kill shot)",
        tol=1e-11, steps=80, ds_list=(0.25,), signs=(+1.0,), grid=GRID_COARSE)

    log("")
    log("=" * 78)
    log("SCALING CHECK (Test A vs Test B, sign=+1, ds=0.25):")
    log("=" * 78)
    a = res_tight.get((+1.0, 0.25))
    b = res_loose.get((+1.0, 0.25))
    if a is not None and b is not None:
        ratio_res = fmax0_loose / max(fmax0_tight, 1e-300)
        log(f"root ||F||_max ratio  (loose/tight) = {ratio_res:.3e}")
        log("PASS criterion: movement ratio should track (be of comparable")
        log("order to, not wildly exceed) the residual ratio -- movement")
        log("that is O(1) regardless of root quality would be the DAE-leakage")
        log("signature (constraint/mass-term touching non-evolution rows).")
        # BUG FIX (M1 fix round 1): a["abs_move"] is a legitimate EXACT 0.0
        # whenever step() short-circuits at k=0 -- i.e. the tight root's own
        # ||F||_max already sits below step()'s Newton tol=1e-10, so no
        # correction is ever computed and z is returned byte-identical to
        # z_old. That is the T4 fix working, not an edge case to paper over.
        # A ratio with that as its denominator (previously max(a, 1e-300))
        # manufactures a ~1e296 number and trips the flag below on every
        # run where the tight root is this good, contradicting the VERDICT
        # section a few lines down which checks the same quantity correctly
        # (against an absolute, residual-scaled bar). Below a fixed noise
        # floor the ratio is undefined, not large, so report it as such and
        # fall back to the same absolute-bar comparison the VERDICT uses --
        # no change to march_s.py, no eigenvalues, no new projector.
        MOVE_FLOOR = 1e-13
        if a["abs_move"] < MOVE_FLOOR:
            bar_abs = max(fmax0_loose * 0.25 * 50.0, 1e-8)
            log(f"step |z_new-z_old|_max (tight) = {a['abs_move']:.3e} -- below "
                f"the {MOVE_FLOOR:.0e} noise floor (step() returned at k=0, "
                "no correction computed); a loose/tight RATIO is undefined "
                "here, not large. Falling back to an absolute bar on the "
                "loose-root movement instead:")
            log(f"  loose |z_new-z_old|_max = {b['abs_move']:.3e}  vs  "
                f"bar = {bar_abs:.3e}")
            if b["abs_move"] > bar_abs:
                log(">>> FLAG: loose-root movement exceeds the residual-scaled "
                    "bar -- possible leakage, investigate before trusting T4 fix.")
            else:
                log(">>> loose-root movement is within the residual-scaled bar "
                    "-- no leakage signature at this scale.")
        else:
            ratio_move = b["abs_move"] / a["abs_move"]
            log(f"step |z_new-z_old|_max ratio (loose/tight) = {ratio_move:.3e}")
            if ratio_move > 1e3 * max(ratio_res, 1.0):
                log(">>> FLAG: movement ratio wildly exceeds residual ratio -- "
                    "possible leakage, investigate before trusting T4 fix.")
            else:
                log(">>> movement ratio is consistent with (not wildly exceeding) "
                    "the residual ratio -- no leakage signature at this scale.")
    else:
        log("one of the two step()s FAILED -- cannot form the ratio; see above.")

    log("")
    log("=" * 78)
    log("VERDICT (Test A, the actual kill shot, sign=+1.0 S4-structural, ds=0.25):")
    log("=" * 78)
    a025 = res_tight.get((+1.0, 0.25))
    if a025 is None:
        log("FAIL -- step() returned None (linear solve broke) at the fixed point.")
    else:
        # PASS bar: movement should sit at/near the root's own residual x ds
        # scale, not O(1) of the state norm.
        bar = max(fmax0_tight * 0.25 * 50.0, 1e-8)  # generous 50x safety factor
        verdict = "PASS" if a025["abs_move"] < bar else "FAIL"
        log(f"root ||F||_max = {fmax0_tight:.3e}   step |z_new-z_old|_max = "
            f"{a025['abs_move']:.3e}   generous bar = {bar:.3e}")
        log(f"{verdict}: fixed point stays fixed to within a residual-scaled "
            f"bar under one backward-Euler step." if verdict == "PASS" else
            f"{verdict}: fixed point MOVES more than the root's own residual "
            f"can explain -- constraint/mass-term leakage.")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
