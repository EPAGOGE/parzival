"""
Extract the GEOMETRY of the omega1 peak from saved checkpoints, to decide whether
two runs sit on the SAME self-similar fixed point.

WHY t_s CANNOT ANSWER THAT
--------------------------
t_s moves continuously with the initial data -- amplitude, sign, profile all shift
it. A 30% change in t_s between two ICs is therefore NO evidence of a different
blowup class. What IS class-invariant:

  1. the argmax LOCATION -- on the corner ring (z*=0, r*=1) or off it. Boussinesq
     s=4 drifted off the corner (argmax x/pi = 0.93), which is what told us it was
     a different stage of the dynamics, not a different singularity.
  2. the SPATIAL self-similar exponent -- peak width ell(t) ~ (t_s - t)^p. This is
     the genuine eigenvalue (the axisym analogue of Boussinesq's gamma = -c_l/c_w).
     The temporal exponent is FORCED by the scaling group and so carries no
     discriminating information at all: every run reproduces -1 by construction.
  3. the ANISOTROPY ell_z/ell_r -> a constant if the profile has converged.

So: forced temporal law = a consistency check; spatial exponent + location = the
actual discriminator. This script measures 2 and 3 and reports 1.

Reads the engine's own checkpoints via solver.load_state, so the domain, bases and
tau terms are constructed by the SAME code that ran the simulation -- no duplicated
domain logic that could silently disagree.
"""
import argparse
import importlib.util
import pathlib
import sys

import h5py
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "ax", str(pathlib.Path(__file__).with_name("dedalus_axisym.py")))
ax = importlib.util.module_from_spec(_spec)
sys.modules["ax"] = ax
_spec.loader.exec_module(ax)


def half_width(coord, prof, ipk):
    """Half-width at half maximum of |prof| about index ipk, along `coord`.
    Linear interpolation on the first crossing of peak/2 on each side; returns the
    mean of the two sides, or the single available side at a domain edge."""
    pk = prof[ipk]
    if pk <= 0:
        return np.nan
    half = 0.5 * pk
    sides = []
    for step in (+1, -1):
        i = ipk
        while 0 <= i + step < len(prof) and prof[i + step] > half:
            i += step
        j = i + step
        if not (0 <= j < len(prof)):
            continue                      # peak runs into the boundary on this side
        f = (prof[i] - half) / max(prof[i] - prof[j], 1e-300)
        sides.append(abs(coord[i] + f * (coord[j] - coord[i]) - coord[ipk]))
    return float(np.mean(sides)) if sides else np.nan


def wrap(dz, L):
    """Periodic displacement into [-L/2, L/2]."""
    return (dz + 0.5 * L) % L - 0.5 * L


def sym_point(zstar, L):
    """Nearest ZERO of the IC's sin(2 pi z / L) -- i.e. the nearest symmetry point,
    which is where omega1 is odd and where the degeneracy order is defined. |omega1|
    peaks a finite distance AWAY from it, so the order must be fitted near the zero,
    not near the peak."""
    cands = np.array([0.0, 0.5 * L, L])
    return float(cands[np.argmin(np.abs(cands - zstar))])


def collapse(snaps, z, L, frac=0.5, nxi=61, xi_max=3.0):
    """SELF-SIMILAR PROFILE COLLAPSE -- the strongest discriminator available here.

    A single exponent is one number; the profile is the whole function. If the run
    is on a self-similar fixed point, then plotting

        omega1(z, r*) / sup|omega1|     against     xi = z / ell_z(t)

    must collapse EVERY late snapshot onto one master curve. Returns (xi, master,
    spread) where spread is the mean over xi of the max-min across snapshots --
    small spread = converged profile. Comparing master curves ACROSS lattice points
    is then a direct test of same-fixed-point, far sharper than comparing p or t_s.
    """
    late = snaps[int(len(snaps) * (1 - frac)):]
    xi = np.linspace(-xi_max, xi_max, nxi)
    curves = []
    for s_ in late:
        prof, ell, pk, zstar = s_[10], s_[3], s_[5], s_[1]
        if not (np.isfinite(ell) and ell > 0 and pk > 0):
            continue
        # CENTER on the peak with periodic wrapping. Without this the test compares
        # peaks sitting at different z to each other and is meaningless.
        dz = wrap(z - zstar, L)
        o = np.argsort(dz)
        curves.append(np.interp(xi, dz[o] / ell, (prof / pk)[o]))
    if len(curves) < 3:
        return xi, None, np.nan
    C = np.array(curves)
    master = C.mean(axis=0)
    spread = float(np.mean(C.max(axis=0) - C.min(axis=0)))
    return xi, master, spread


def z_order_scaled(z, prof, L, zsym, ell, lo=0.03, hi=0.35):
    """Leading power of |prof| about the SYMMETRY POINT zsym, on a window that SCALES
    with the current peak width ell: |dz| in [lo*ell, hi*ell]. A fixed window in z is
    wrong once the structure sharpens past it -- it then measures the tail rather than
    the dz -> 0 behaviour, and reports a spurious order near 0."""
    if not (np.isfinite(ell) and ell > 0):
        return np.nan, np.nan
    dz = np.abs(wrap(z - zsym, L))
    m = (dz > lo * ell) & (dz < hi * ell) & (prof > 0)
    if m.sum() < 5:
        return np.nan, np.nan
    x, y = np.log(dz[m]), np.log(prof[m])
    c = np.polyfit(x, y, 1)
    r2 = 1 - ((y - np.polyval(c, x)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return float(c[0]), float(r2)


def z_order(z, prof, L, lo=0.004, hi=0.06):
    """Leading power of |prof| in z near z=0: prof ~ z^p, log-log fit on the wall ray.
    Used to verify a run STAYS at its lattice point (q, p) all the way to t ~ t_s,
    rather than drifting -- the lattice label is a claim about the whole run, not
    just about the initial data."""
    m = (z > lo * L) & (z < hi * L) & (prof > 0)
    if m.sum() < 6:
        return np.nan, np.nan
    x, y = np.log(z[m]), np.log(prof[m])
    c = np.polyfit(x, y, 1)
    r2 = 1 - ((y - np.polyval(c, x)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return float(c[0]), float(r2)


def snapshots(ckpt_dir, Nz, Nr, r0=0.4):
    """Yield (t, z*, r*, ell_z, ell_r, sup|w1|) for every write in every h5, in
    time order. Fields are read at scale 1 -- safe here because nothing is stepping."""
    files = sorted(pathlib.Path(ckpt_dir).glob("*.h5"))
    if not files:
        raise SystemExit(f"no checkpoints in {ckpt_dir}")
    b = ax.build(Nz, Nr, r0=r0)
    _, solver = ax.make_ivp(b)
    z = np.asarray(b.z).ravel()
    r = np.asarray(b.r).ravel()
    out = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            n = f["scales/sim_time"].shape[0]
        for k in range(n):
            solver.load_state(str(fp), k)
            b.omega1.change_scales(1)
            w = np.abs(np.asarray(b.omega1["g"]))
            if w.shape != (z.size, r.size):
                raise SystemExit(f"grid mismatch {w.shape} vs {(z.size, r.size)}")
            i, j = np.unravel_index(int(np.argmax(w)), w.shape)
            b.u1.change_scales(1)
            u = np.abs(np.asarray(b.u1["g"]))
            jw = int(np.argmax(r))                       # wall ring
            _ell = half_width(z, w[:, j], i)
            _zs = sym_point(float(z[i]), b.L)
            pw, pw_r2 = z_order_scaled(z, w[:, jw], b.L, _zs, _ell)
            pu, pu_r2 = z_order_scaled(z, u[:, jw], b.L, _zs, _ell)
            out.append((float(solver.sim_time), float(z[i]), float(r[j]),
                        half_width(z, w[:, j], i), half_width(r, w[i, :], j),
                        float(w[i, j]), pu, pw, pu_r2, pw_r2, w[:, j].copy()))
    out.sort(key=lambda e: e[0])
    return out, b.L, z


def ts_from_forced_law(t, w, frac=0.40):
    """t_s from the theory-forced omega1 ~ (t_s-t)^-1: 1/w is linear, root at t_s."""
    t = np.asarray(t, float); w = np.asarray(w, float)
    k = int(len(t) * (1 - frac))
    t, w = t[k:], w[k:]
    if len(t) < 3:
        return np.nan                    # too few snapshots to locate t_s
    p = np.polyfit(t, 1.0 / w, 1)
    return -p[1] / p[0]


def fit_exponent(t, ell, ts, frac=0.5, ell_min=0.0):
    """ell ~ (t_s - t)^p over the late window; returns (p, R^2, npts).

    ell_min DISCARDS snapshots whose measured width is at or below a few grid cells.
    Without it the fit is contaminated by exactly the points where the width is no
    longer measurable: at Nz=256 the z spacing is L/256 = 6.5e-4 and the peak
    half-width reaches ~5e-4 by t=0.0034 -- BELOW ONE CELL. A width smaller than the
    grid is a resolution artefact, not a measurement."""
    t = np.asarray(t, float); ell = np.asarray(ell, float)
    if not np.isfinite(ts):
        return np.nan, np.nan, 0
    m = np.isfinite(ell) & (ell > ell_min) & (t < ts)
    t, ell = t[m], ell[m]
    k = int(len(t) * (1 - frac))
    t, ell = t[k:], ell[k:]
    if len(t) < 4:
        return np.nan, np.nan, len(t)
    x, y = np.log(ts - t), np.log(ell)
    p = np.polyfit(x, y, 1)
    r2 = 1 - ((y - np.polyval(p, x)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return float(p[0]), float(r2), len(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run-ids whose ckpt_<id> dirs to analyse")
    ap.add_argument("--Nz", type=int, default=256)
    ap.add_argument("--Nr", type=int, default=768)
    ap.add_argument("--root", default=str(pathlib.Path.home() / "parzival/runs"))
    a = ap.parse_args()

    print("PEAK GEOMETRY -- the class discriminator (t_s is NOT one)\n")
    masters, _Z = {}, {}
    for rid in a.runs:
        snaps, L, zgrid = snapshots(pathlib.Path(a.root) / f"ckpt_{rid}", a.Nz, a.Nr)
        _Z[rid] = zgrid
        t = [s[0] for s in snaps]; w = [s[5] for s in snaps]
        ts = ts_from_forced_law(t, w)
        dz_grid = L / a.Nz
        dr_grid = 0.6 / a.Nr          # annulus width / Nr (Chebyshev: coarsest cell)
        ELL_CELLS = 4.0
        pz, r2z, nz = fit_exponent(t, [s[3] for s in snaps], ts,
                                   ell_min=ELL_CELLS * dz_grid)
        pr, r2r, nr = fit_exponent(t, [s[4] for s in snaps], ts,
                                   ell_min=ELL_CELLS * dr_grid)
        nres = sum(1 for s in snaps if np.isfinite(s[3]) and s[3] <= ELL_CELLS * dz_grid)
        print(f"  z grid spacing = {dz_grid:.3e};  {nres}/{len(snaps)} snapshots have "
              f"ell_z <= {ELL_CELLS:g} cells and are DISCARDED as unresolved")
        last = snaps[-1]
        print(f"=== {rid} ===   {len(snaps)} snapshots, t_s(forced) = {ts:.6f}")
        # |omega1| peaks at symmetry-EQUIVALENT z (z=0 and z=L/2 are both zeros of the
        # IC), so a raw z*/L jumps between them and means nothing. Report the distance
        # to the nearest symmetry point instead, plus r*.
        dsym = [abs(wrap(e[1] - sym_point(e[1], L), L)) / L for e in snaps[-6:]]
        print(f"  r* at last 6 snapshots: "
              f"{', '.join(f'{e[2]:.5f}' for e in snaps[-6:])}   (wall is r*=1)")
        print(f"  |z*-z_sym|/L  last 6 : "
              f"{', '.join(f'{v:.4f}' for v in dsym)}   (corner ring is 0)")
        print(f"  ell_z ~ (t_s-t)^{pz:+.4f}   R2={r2z:.5f}  [{nz} pts]")
        print(f"  ell_r ~ (t_s-t)^{pr:+.4f}   R2={r2r:.5f}  [{nr} pts]")
        aniso = [s[3] / s[4] for s in snaps if np.isfinite(s[3]) and np.isfinite(s[4])
                 and s[4] > 0]
        if len(aniso) >= 4:
            tail = aniso[len(aniso) // 2:]
            print(f"  anisotropy ell_z/ell_r: last-half mean {np.mean(tail):.4f}"
                  f"  spread {100*(max(tail)-min(tail))/np.mean(tail):.1f}%"
                  f"   (constant => profile converged)")
        print(f"  lattice label at LAST snapshot: ord_z u1 = {last[6]:+.3f}"
              f" (R2={last[8]:.4f})   ord_z w1 = {last[7]:+.3f} (R2={last[9]:.4f})")
        print("  t         z*/L      r*        ell_z      ell_r     sup|w1|"
              "    ord u1  ord w1")
        for s in snaps[-6:]:
            print(f"  {s[0]:.6f}  {s[1]/L:8.5f}  {s[2]:.6f}  {s[3]:9.3e}  "
                  f"{s[4]:9.3e}  {s[5]:9.4g}  {s[6]:+7.3f} {s[7]:+7.3f}")
        xi, master, spread = collapse(snaps, _Z[rid], L)
        if master is not None:
            print(f"  PROFILE COLLAPSE over last-half snapshots: spread = {spread:.4f}"
                  f"   ({'CONVERGED' if spread < 0.05 else 'NOT converged'})")
            masters[rid] = (xi, master)
        print()

    if len(masters) > 1:
        print("=== CROSS-RUN PROFILE COMPARISON (same fixed point <=> same curve) ===")
        ks = list(masters)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a_, b_ = masters[ks[i]][1], masters[ks[j]][1]
                d = float(np.mean(np.abs(a_ - b_)))
                print(f"  {ks[i]:5s} vs {ks[j]:5s}: mean |dprofile| = {d:.4f}"
                      f"   {'SAME' if d < 0.05 else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
