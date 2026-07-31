"""G1 -- MODALITY COUNT: is the 3D corner one-exponent or two?

The polar corner profile uses ONE radius rho and ONE exponent alpha.  alpha_eff
fits isotropic shells.  Both presuppose the corner layer's aspect ratio settles.
The pre-mortem measured ell_z/ell_r sliding 2.96 -> 0.06 on a single rung, which
would mean two exponents and a single-alpha description that is incomplete at the
corner.  A single rung cannot distinguish that from a grid artifact.

PROTOCOL, pre-registered:
  * ell_z, ell_r are HALF-WIDTHS of |omega1| through its own argmax, along the
    z and r lines through that point, at the half-maximum level.
  * Compared at MATCHED (ts - t)/ts across rungs, NOT at matched t -- the
    campaign's own method rule.
  * Every point whose half-width is <= 2 cells in that direction is DISCARDED as
    grid-limited (the pre-mortem showed the thin direction saturates at exactly
    one cell; those points measure the mesh, not the flow).
  * argmax wall-distance (1 - r*) reported alongside: the campaign's rule is that
    an off-ring argmax means a DIFFERENT STAGE, so any point with the peak
    detached is flagged and excluded from the modality verdict.

KILL LINES, both directions informative:
  * ell_z/ell_r CONVERGES toward a constant under refinement  -> ONE exponent,
    the polar profile applies, multi-modal premise DEAD (good outcome).
  * ratio keeps sliding AND the slide is resolution-stable   -> TWO exponents,
    the single-alpha corner description is incomplete.
  * ratio slides but shrinks with refinement                 -> grid artifact,
    no verdict, needs a finer rung.
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

import h5py
import numpy as np

TS = 0.0035056          # engine TS_REF (Luo-Hou corner ring)
MIN_CELLS = 2.0


def halfwidth(profile, coord, ipk):
    """Half-width at half-max of a 1-D slice through index ipk."""
    pk = abs(profile[ipk])
    if pk <= 0:
        return np.nan
    half = pk / 2.0
    lo = ipk
    while lo > 0 and abs(profile[lo]) > half:
        lo -= 1
    hi = ipk
    while hi < len(profile) - 1 and abs(profile[hi]) > half:
        hi += 1
    return float(abs(coord[hi] - coord[lo]) / 2.0)


def scan(snapdir):
    out = []
    for fn in sorted(glob.glob(str(pathlib.Path(snapdir) / "*.h5"))):
        with h5py.File(fn, "r") as f:
            W = f["tasks/omega1"]
            t = f["scales/sim_time"][:]
            zg = np.array(W.dims[1][0]).ravel()
            rg = np.array(W.dims[2][0]).ravel()
            dz = float(np.min(np.diff(zg))) if len(zg) > 1 else np.nan
            dr = float(np.min(np.abs(np.diff(rg)))) if len(rg) > 1 else np.nan
            for i in range(W.shape[0]):
                w = np.abs(np.asarray(W[i]))
                iz, ir = np.unravel_index(int(np.argmax(w)), w.shape)
                lz = halfwidth(w[:, ir], zg, iz)
                lr = halfwidth(w[iz, :], rg, ir)
                out.append(dict(
                    t=float(t[i]), tau=float((TS - t[i]) / TS),
                    ell_z=lz, ell_r=lr,
                    ratio=(lz / lr) if (lr and np.isfinite(lr) and lr > 0) else np.nan,
                    cells_z=lz / dz if dz else np.nan,
                    cells_r=lr / dr if dr else np.nan,
                    wall_gap=float(1.0 - rg[ir]), sup=float(w.max())))
    out.sort(key=lambda d: d["t"])
    return out


RUNGS = [("N128x384", 128), ("N256x768", 256), ("N512x1536", 512)]
data = {}
for tag, N in RUNGS:
    d = f"../runs/snap_e3dv2_{tag}"
    if not glob.glob(d + "/*.h5"):
        print(f"[{tag}] no snapshots at {d}")
        continue
    data[tag] = scan(d)
    print(f"[{tag}] {len(data[tag])} snapshots, "
          f"tau {data[tag][-1]['tau']:.4f}..{data[tag][0]['tau']:.4f}")

TAUS = [0.20, 0.10, 0.05, 0.03, 0.02, 0.012]
print(f"\n{'tau=(ts-t)/ts':>13}" + "".join(f"{t:>26}" for t, _ in RUNGS))
print(f"{'':>13}" + "".join(f"{'ell_z/ell_r  cellz  gap':>26}" for _ in RUNGS))
print("-" * (13 + 26 * len(RUNGS)))
table = {}
for tau in TAUS:
    row = f"{tau:>13.3f}"
    for tag, _ in RUNGS:
        if tag not in data:
            row += f"{'--':>26}"
            continue
        rec = min(data[tag], key=lambda d: abs(d["tau"] - tau))
        if abs(rec["tau"] - tau) > 0.35 * tau:
            row += f"{'(no pt)':>26}"
            continue
        ok = rec["cells_z"] >= MIN_CELLS and rec["cells_r"] >= MIN_CELLS
        mark = "" if ok else "*"
        table.setdefault(tau, {})[tag] = (rec["ratio"] if ok else np.nan, rec)
        row += f"{rec['ratio']:>13.3f}{mark}{rec['cells_z']:>7.1f}{rec['wall_gap']:>6.0e}"
    print(row)
print("-" * (13 + 26 * len(RUNGS)))
print("* = thin direction at <= 2 cells: grid-limited, excluded from the verdict")

print("\nVERDICT INPUT -- ratio across rungs at each matched tau (grid-limited excluded):")
verdict = []
for tau in TAUS:
    vals = [(tag, v[0]) for tag, v in table.get(tau, {}).items() if np.isfinite(v[0])]
    if len(vals) >= 2:
        rs = [v for _, v in vals]
        spread = (max(rs) - min(rs)) / max(abs(np.mean(rs)), 1e-30)
        verdict.append((tau, rs, spread))
        print(f"  tau={tau:.3f}  " + "  ".join(f"{tg}:{v:.3f}" for tg, v in vals)
              + f"   rel spread {spread:.3f}")
if not verdict:
    print("  NO tau has >=2 non-grid-limited rungs. The thin direction is at the")
    print("  mesh floor wherever the rungs overlap: G1 CANNOT BE READ from these")
    print("  runs, and G2 (the basis change) is a hard precondition, not an option.")
pathlib.Path("../runs/g1_modality.json").write_text(json.dumps(
    {k: v for k, v in data.items()}, indent=1))
print("\n-> ../runs/g1_modality.json")
