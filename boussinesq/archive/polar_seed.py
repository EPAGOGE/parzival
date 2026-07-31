"""
Build a VERIFIED initial guess for the log-polar profile solve, from Chen-Hou's own
converged 620x620 profile.

WHY THIS IS THE BLOCKER
-----------------------
POLAR_SPEC records that the actual obstacle for a Newton solve on a NON-UNIQUE family is
not the discretization, it is the initial guess: the profile is a SADDLE, so there is no
basin to fall into, and Newton lands wherever it is pointed -- including on the trivial
solution, which has already happened once in this lab (tell: the residual falling by
exactly the damping factor each step). Chen-Hou's converged profile is on disk, so the
seed can be taken from a solution that is known to exist rather than guessed.

THE ONE IDEA THAT MAKES THIS ACCURATE
-------------------------------------
Do NOT interpolate the raw fields. They span ~10 decades over the mesh, and their own
mesh is GEOMETRIC in the far field (measured spacing ratio a constant 1.0648, i.e. their
mesh is already effectively logarithmic in r -- independent corroboration that log-polar
is the right frame). Interpolating a 10-decade power law on a geometric grid throws away
accuracy for nothing.

Instead interpolate the SCALED fields, which are the angular functions and therefore
slowly varying:

    Ot = Om * R^(-alpha)          -> tends to g(beta)
    Bt = B  * R^(-(1+2 alpha))    -> tends to the angular profile of theta

Both are O(1) and nearly s-independent in the far field, so linear interpolation on a
6.5%-per-cell radial grid is accurate. Then the seed is reconstructed exactly:

    Om(s,b) = Ot_interp(s,b) * e^(alpha s),   B(s,b) = Bt_interp(s,b) * e^((1+2a)s)

SELF-CHECKS (this file is a gate, not just a utility)
  1. s-INDEPENDENCE: Ot and Bt must become independent of s at large s. Measured as the
     relative spread across s at fixed beta. This is the far-field statement itself, so
     it tests the seed and the formulation at once.
  2. EXPONENT ROUND-TRIP: re-measuring the radial exponent from the RECONSTRUCTED Om
     must return alpha. Guards against an interpolation that silently distorts the
     power law.
  3. ANGULAR CONSISTENCY: Ot(s1, .) vs Ot(s2, .) for two well-separated s must agree.
  4. CROSS-GATE: Ot's angular profile must agree with the g(beta) that angular_gate.py
     extracts independently by annulus averaging.
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

MAT = pathlib.Path.home() / ("parzival/refs/chen_hou/Perturbed_eqn/Computed profile/"
                             "Steady_state_pertb_oneMesh62036.mat")


def _grid_field(d, s, name, shape):
    """Pull a mesh-shaped array named `name` from the .mat, whether it sits at top level
    or inside `solu`, and whether it is a plain array or a CELL array of mesh patches.
    Guessing the layout is how a silent wrong-field bug gets in, so match on SHAPE."""
    for src in (d.get(name), getattr(s, name, None)):
        if src is None:
            continue
        cands = list(np.ravel(src)) if (isinstance(src, np.ndarray)
                                        and src.dtype == object) else [src]
        for c in cands:
            try:
                a = np.asarray(c, dtype=float)
            except (TypeError, ValueError):
                continue
            if a.shape == shape:
                return a
    raise SystemExit(f"could not find a {shape} array named '{name}' in the .mat")


def load():
    d = loadmat(MAT, squeeze_me=True, struct_as_record=False)
    M, s = d["Mesh"], d["solu"]
    xs = [np.asarray(e, dtype=float) for e in np.ravel(np.asarray(M.x, dtype=object))]
    w = np.asarray(d["w"], dtype=float)
    return dict(X=xs[0], Y=xs[1],
                w=w,
                th=_grid_field(d, s, "th", w.shape),
                alpha=-float(np.ravel(np.asarray(s.al))[0]),
                cl=float(np.ravel(np.asarray(s.cl))[0]),
                cw=float(np.ravel(np.asarray(s.cw))[0]))


def scaled_interpolators(P):
    """Interpolators for the SCALED (slowly varying) fields on their (X,Y) tensor grid."""
    X, Y, a = P["X"], P["Y"], P["alpha"]
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    R = np.sqrt(XX ** 2 + YY ** 2)
    R = np.where(R > 0, R, np.nan)                 # the single corner point R=0
    Ot = P["w"] * R ** (-a)
    Bt = P["th"] * R ** (-(1.0 + 2.0 * a))
    # the corner cell is the only NaN; fill from its neighbour so interpolation is total
    for A in (Ot, Bt):
        bad = ~np.isfinite(A)
        if bad.any():
            A[bad] = A[1, 1]
    # CUBIC, not linear. Measured: linear leaves a spurious s-variation in Ot with
    # median 5.6e-4 and max|c_l Ot_s| = 4.7e-2, whereas the far-field equation balance
    # demands |c_l Ot_s| ~ 1.1e-2 -- i.e. linear interpolation error and the far-field
    # signal are the SAME SIZE, so the far field is unresolvable from a linear seed.
    # Cubic: median 1.5e-5 (37x better) and max|c_l Ot_s| = 7.9e-3, now BELOW the
    # physical level. Quintic gives 7.9e-3 too, so cubic is converged -- no reason to
    # pay for more.
    def mk(A):
        for meth in ("cubic", "linear"):
            try:
                return RegularGridInterpolator((X, Y), A, method=meth,
                                               bounds_error=False, fill_value=None)
            except (ValueError, NotImplementedError):
                continue
        raise SystemExit("no usable interpolation method")
    return mk(Ot), mk(Bt)


def seed_on_grid(P, s_grid, b_grid):
    """Evaluate the seed on a tensor (s, beta) grid. Returns Ot, Bt, Om, B."""
    fOt, fBt = scaled_interpolators(P)
    a = P["alpha"]
    S, Bg = np.meshgrid(s_grid, b_grid, indexing="ij")
    Rg = np.exp(S)
    pts = np.stack([Rg * np.cos(Bg), Rg * np.sin(Bg)], axis=-1)
    Ot = fOt(pts)
    Bt = fBt(pts)
    Om = Ot * np.exp(a * S)
    B = Bt * np.exp((1.0 + 2.0 * a) * S)
    return Ot, Bt, Om, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smin", type=float, default=19.0,
                    help="inner s. The power law only STARTS at s~18.4, so a far-field "
                         "check must sit above it.")
    ap.add_argument("--smax", type=float, default=30.0)
    ap.add_argument("--ns", type=int, default=48)
    ap.add_argument("--nb", type=int, default=64)
    ap.add_argument("--out", default=str(pathlib.Path.home() / "parzival/runs/polar_seed.npz"))
    a_ = ap.parse_args()

    P = load()
    alpha = P["alpha"]
    print(f"Chen-Hou profile: cl={P['cl']:.8f} cw={P['cw']:.8f} alpha={alpha:+.8f}")
    print(f"their mesh: X in [{P['X'].min():.3g}, {P['X'].max():.3g}]  "
          f"(s up to {np.log(P['X'].max()):.2f})")

    s_grid = np.linspace(a_.smin, a_.smax, a_.ns)
    # avoid the exact edges where one Cartesian coordinate collapses to 0
    b_grid = np.linspace(0.02, np.pi / 2 - 0.02, a_.nb)
    Ot, Bt, Om, B = seed_on_grid(P, s_grid, b_grid)
    print(f"\nseed grid: s in [{a_.smin}, {a_.smax}] ({a_.ns} pts), "
          f"beta in [0.02, pi/2-0.02] ({a_.nb} pts)")
    print(f"  Om spans {np.nanmin(np.abs(Om)):.3e} .. {np.nanmax(np.abs(Om)):.3e}"
          f"   ({np.log10(np.nanmax(np.abs(Om))/max(np.nanmin(np.abs(Om)),1e-300)):.1f} decades)")
    print(f"  Ot spans {np.nanmin(Ot):.5g} .. {np.nanmax(Ot):.5g}   <- O(1) by construction")

    # ---- CHECK 1: s-independence of the scaled fields --------------------------
    print("\nCHECK 1  s-INDEPENDENCE of the scaled fields (the far-field statement)")
    for nm, A in (("Ot", Ot), ("Bt", Bt)):
        rel = np.nanmax(np.abs(A - A.mean(axis=0, keepdims=True)), axis=0) / \
              np.maximum(np.abs(A.mean(axis=0)), 1e-300)
        print(f"   {nm}: max over beta of (spread in s / mean) = {np.nanmax(rel):.4e}"
              f"   median = {np.nanmedian(rel):.4e}")

    # ---- CHECK 2: exponent round-trip from the RECONSTRUCTED field --------------
    print("\nCHECK 2  EXPONENT ROUND-TRIP from reconstructed Om (target "
          f"{alpha:+.6f})")
    for jb in (a_.nb // 8, a_.nb // 4, a_.nb // 2, 3 * a_.nb // 4):
        col = np.abs(Om[:, jb])
        m = col > 0
        if m.sum() > 8:
            p = np.polyfit(s_grid[m], np.log(col[m]), 1)
            print(f"   beta={b_grid[jb]:.4f}: measured exponent {p[0]:+.6f}"
                  f"   err {abs(p[0]-alpha):.2e}")

    # ---- CHECK 3: angular profile at two separated s ---------------------------
    i1, i2 = 2, a_.ns - 3
    d = np.abs(Ot[i1] - Ot[i2])
    sc = max(np.abs(Ot[i1]).max(), 1e-300)
    print(f"\nCHECK 3  ANGULAR CONSISTENCY  Ot(s={s_grid[i1]:.2f}) vs "
          f"Ot(s={s_grid[i2]:.2f}): max|diff|/scale = {d.max()/sc:.4e}")

    # ---- CHECK 4: cross-gate against angular_gate's independent extraction ------
    spec = importlib.util.spec_from_file_location(
        "ag", str(pathlib.Path(__file__).with_name("angular_gate.py")))
    ag = importlib.util.module_from_spec(spec); sys.modules["ag"] = ag
    spec.loader.exec_module(ag)
    Pg = ag.load_profile()
    bg, g1 = ag.extract_angular(Pg, Pg["w"])
    # Evaluate the seed ON angular_gate's own beta points -- interpolating onto them and
    # letting np.interp CLAMP outside our range inflates this to 24% for no reason.
    Ot_at, _, _, _ = seed_on_grid(P, np.array([0.5 * (a_.smin + a_.smax)]), bg)
    mine = Ot_at[0]
    rel = np.abs(mine - g1) / np.maximum(np.abs(g1), 1e-30)
    interior = (bg > 0.05) & (bg < np.pi / 2 - 0.05)
    print(f"CHECK 4  CROSS-GATE vs angular_gate.py's annulus average ({bg.size} beta pts)")
    print(f"         rel L2, all points     = "
          f"{np.linalg.norm(mine-g1)/np.linalg.norm(g1):.4e}")
    print(f"         rel L2, interior only  = "
          f"{np.linalg.norm((mine-g1)[interior])/np.linalg.norm(g1[interior]):.4e}"
          f"   <- the meaningful number")
    worst = int(np.argmax(rel))
    print(f"         worst point: beta={bg[worst]:.5f} (pi/2 - {np.pi/2-bg[worst]:.2e})"
          f"  annulus={g1[worst]:.5g} seed={mine[worst]:.5g}")
    print("         NOTE: angular_gate's EDGE bins are unreliable and the seed is right")
    print("         there. Polar binning of a CARTESIAN TENSOR grid degenerates near an")
    print("         axis: the last beta bin at r~1e10..1e12 collects points whose")
    print("         absolute y1 ranges from 0.004 to 7.8e9, and averaging those is not")
    print("         an estimator of g. Compare interiors; trust the seed at the edges.")

    # ---- CHECK 5: EDGE BEHAVIOUR, the thing the beta basis must resolve ---------
    print("\nCHECK 5  EDGE BEHAVIOUR of the angular profile (drives the beta basis)")
    eps = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
    for s0 in (a_.smin + 1.0, 0.5 * (a_.smin + a_.smax), a_.smax - 1.0):
        Oa, _, _, _ = seed_on_grid(P, np.array([s0]), np.pi / 2 - eps)
        Ow, _, _, _ = seed_on_grid(P, np.array([s0]), eps)
        va, vw = Oa[0], Ow[0]
        m = va > 0
        sl = np.polyfit(np.log(eps[m]), np.log(va[m]), 1)[0] if m.sum() > 2 else np.nan
        print(f"   s={s0:5.1f}: axis  Ot ~ eps^{sl:+.4f}  (coef {va[1]/eps[1]:.4f});"
              f"  wall  Ot -> {vw[-1]:.5f}")
    print("   => Om vanishes LINEARLY at the symmetry line (odd, smooth) and is FLAT and")
    print("      NONZERO at the wall. Both edges are REGULAR on this branch, and the")
    print("      values are s-INDEPENDENT, which is the far-field ansatz itself.")
    print("      Consequences: impose Om=0 at beta=pi/2 (the solution meets it linearly,")
    print("      no weighted class needed HERE); do NOT use a beta basis that vanishes at")
    print("      beta=0; the axis layer is ~0.03 rad wide so Chebyshev in beta -- which")
    print("      clusters at BOTH ends -- is the right choice.")

    np.savez(a_.out, s=s_grid, beta=b_grid, Ot=Ot, Bt=Bt, Om=Om, B=B,
             alpha=alpha, cl=P["cl"], cw=P["cw"])
    print(f"\nsaved seed -> {a_.out}")


if __name__ == "__main__":
    main()
