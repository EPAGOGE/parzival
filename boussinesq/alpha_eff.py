"""alpha_eff(t): the SPATIAL self-similar exponent from march snapshots.

Instrument for the e3-physical-run drift test.  The profile campaign established
Omega ~ rho^alpha in the corner-profile far field (verified pointwise, kappa=1).
Made physical: at each snapshot, in the corner frame, the vorticity amplitude
across the self-similar window scales as a power of distance-from-corner, and
that power is alpha_eff(t).  No T-estimate enters the VALUE (only the abscissa),
which is tension #45 axis 1's mitigation.

PRE-REGISTERED WINDOW RULE (#45 axis 3), fixed before any fit:
  the window is [w_in, w_out] * ell(t), where ell(t) is the collapsing CORNER
  scale = the radius at which the corner-centred shell amplitude peaks, with
  w_in = 3 (above grid scale) and w_out = 30 (below IC scale).
  REVISION 2026-07-28, forced by measurement: the original proxy
  ell = sup|w|/sup|grad w| is GLOBAL and was measured FROZEN at 1.302e-3 across a
  window where (T-t) spans 70x -- pinned by the IC layer, not the corner. It held
  the fit windows off the collapse and manufactured a monotone alpha drift
  (-0.21 -> -0.46) that the two-window replication correctly flagged as artifact.  The fit is
  REPLICATED on a second window [4, 20] and both are reported; a drift that
  appears in one and not the other is a window artifact, not physics.

Usage:  run_dedalus.sh alpha_eff.py --snapdir ../runs/snap_e3c512s [--geom bsq]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import h5py
import numpy as np

WINDOWS = {"main": (3.0, 30.0), "check": (4.0, 20.0)}


def corner_radius(x, z, geom):
    """Distance from the corner, in the geometry the engine used.

    bsq:  corner at (x, z) = (pi, 0), wall at z = 0.
    axi:  corner at (z, r) = (0, 1), wall at r = 1.
    """
    if geom == "bsq":
        # CORNER IS AT x = 0, NOT x = pi.  dedalus_bsq.py contradicts itself
        # (line 14 "x=0 corner symmetry" vs line 115 "corner sits at x=pi"); the
        # DATA settles it: on the wall row the argmax|w| sits at x = 6.185
        # (= -0.098 mod 2pi), while |w| is 1.9e-10 at x=0 and 1.7e-13 at x=pi
        # (both parity zeros, w odd).  Centring at pi put every fit a half period
        # from the structure and froze ell at 3.168 ~ pi -- the bug's signature.
        # Periodic distance, so the corner is approached from both sides.
        dxp = np.minimum(np.abs(x), 2.0 * np.pi - np.abs(x))
        X, Z = np.meshgrid(dxp, z, indexing="ij")
    else:
        X, Z = np.meshgrid(x, 1.0 - z, indexing="ij")
    return np.sqrt(X ** 2 + Z ** 2)


def fit_alpha(rho, amp, lo, hi):
    """Log-log slope of the vorticity amplitude over [lo, hi]."""
    m = (rho >= lo) & (rho <= hi) & (amp > 0) & np.isfinite(amp)
    if m.sum() < 8:
        return np.nan, int(m.sum())
    return float(np.polyfit(np.log(rho[m]), np.log(amp[m]), 1)[0]), int(m.sum())


def process(path, geom):
    out = []
    with h5py.File(path, "r") as f:
        t = f["scales/sim_time"][:]
        wkey = "tasks/w" if "tasks/w" in f else "tasks/omega1"
        xk = [k for k in f["scales"] if k.startswith("x_hash")
              or k.startswith("z_hash") or k.startswith("kr_hash")]
        W = f[wkey]
        # grids: Dedalus writes them as dimension scales on the task
        dims = W.dims
        g0 = np.array(dims[1][0]).ravel() if len(dims) > 1 and len(dims[1]) else None
        g1 = np.array(dims[2][0]).ravel() if len(dims) > 2 and len(dims[2]) else None
        for i in range(W.shape[0]):
            w = np.asarray(W[i])
            if g0 is None or g1 is None:
                continue
            rho = corner_radius(g0, g1, geom)
            aw = np.abs(w)
            supw = aw.max()
            # ell from the field's own scales: sup|w| / sup|grad w| via finite diffs
            gx = np.gradient(aw, g0, axis=0)
            gz = np.gradient(aw, g1, axis=1)
            supg = np.sqrt(gx ** 2 + gz ** 2).max()
            ell = supw / max(supg, 1e-300)
            # radial amplitude profile: max over angle in log-spaced rho shells
            rr = rho.ravel()
            aa = aw.ravel()
            edges = np.logspace(np.log10(max(rr[rr > 0].min(), 1e-12)),
                                np.log10(rr.max()), 60)
            idx = np.digitize(rr, edges)
            rc, ac = [], []
            for k in range(1, len(edges)):
                s = idx == k
                if s.sum() >= 4:
                    rc.append(np.sqrt(edges[k - 1] * edges[k]))
                    ac.append(aa[s].max())
            rc, ac = np.array(rc), np.array(ac)
            # ell MUST track the collapsing CORNER scale.  The global proxy
            # sup|w|/sup|grad w| is pinned by the IC layer (measured FROZEN at
            # 1.302e-3 over a window where (T-t) spans 70x, which forced the
            # windows off the collapse and manufactured a monotone alpha drift).
            # Corner-centred replacement: the radius at which the corner-centred
            # shell amplitude peaks.  ell_glob kept for the record.
            ell_corner = float(rc[int(np.argmax(ac))]) if len(rc) else np.nan
            rec = {"t": float(t[i]), "sup_w": float(supw),
                   "ell": ell_corner, "ell_glob": float(ell)}
            for nm, (a, bnd) in WINDOWS.items():
                al, n = fit_alpha(rc, ac, a * ell_corner, bnd * ell_corner)
                rec[f"alpha_{nm}"] = al
                rec[f"n_{nm}"] = n
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapdir", required=True)
    ap.add_argument("--geom", default="bsq", choices=["bsq", "axi"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(str(pathlib.Path(a.snapdir) / "*.h5")))
    if not files:
        raise SystemExit(f"no .h5 in {a.snapdir}")
    recs = []
    for p in files:
        recs += process(p, a.geom)
    recs.sort(key=lambda r: r["t"])
    print(f"{'t':>9} {'sup_w':>11} {'ell':>10} {'a_main':>9} {'n':>5} {'a_chk':>9}")
    for r in recs:
        print(f"{r['t']:9.5f} {r['sup_w']:11.4e} {r['ell']:10.3e} "
              f"{r['alpha_main']:9.4f} {r['n_main']:5d} {r['alpha_check']:9.4f}")
    dest = a.out or str(pathlib.Path(a.snapdir).parent /
                        f"alphaeff_{pathlib.Path(a.snapdir).name}.json")
    pathlib.Path(dest).write_text(json.dumps(recs, indent=1))
    print(f"\n-> {dest}  ({len(recs)} snapshots)")


if __name__ == "__main__":
    main()
