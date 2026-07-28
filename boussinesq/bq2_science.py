#!/usr/bin/env python
"""Does the WALL geometry blow up? The finding-relevant run for the 2D branch.

The parity box (bq.py) DEcelerated -- but it is the EXCLUDED symmetry class
(theta,w odd across the wall; Chen-Hou proved global existence there). bq2 is
the correct class (theta,w FREE on the wall) with a structurally-faithful
Luo-Hou IC (verified: theta ~x^2 on axis, nonzero on wall, wall-concentrated).
We only ever ran first-light (A=1, t=3). This drives it hard and reads the
singularity-approach diagnostics under a convergence-based trust tier.

SOLVER TRUST TIER (the 2D analog of the swarm's shadow auditor): for a single
deterministic trajectory, independent-recompute redundancy IS convergence
testing -- a growth signal is trusted only if it survives resolution doubling
(spatial) with budgets conserved and the tail wire green. bq2's RK4 temporal
order is already gate-certified (G4b = 4.000), so the spatial ladder + budgets
+ tail wire are the trust stack.

DIAGNOSTICS (all bounded by meter-escape status -- never a bare blowup claim):
  accel   = log-slope of sup|grad theta| over the last quarter vs the prior
            quarter of the TRUSTED window. >1 accelerating (blowup-like);
            <1 decelerating (global-like). Same discriminant as the 1D dial.
  Tstar   = linear extrapolation of 1/sup|w| -> 0 (the proven (T-t)^-1 form);
            reported with fit R^2. A finite positive Tstar just past the window
            with clean fit is the pre-registered signature.
  itgth   = int sup|grad theta| dt (the rigor-backed continuation integral,
            Chae-Nam / Elgindi-Jeong) -- divergence is the criterion, not BKM.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np

HERE = pathlib.Path(__file__).parent
PY = str(HERE.parent / ".venv" / "bin" / "python")


def run_bq2(A: float, nx: int, ny: int, tmax: float) -> dict:
    out = HERE.parent / "runs" / f"bq2sci_A{A:g}_N{nx}.json"
    r = subprocess.run(
        [PY, str(HERE / "bq2.py"), "--scenario", "--A", str(A),
         "--nx", str(nx), "--ny", str(ny), "--tmax", str(tmax), "--out", str(out)],
        cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bq2 A={A} N={nx} failed:\n{r.stdout[-500:]}\n{r.stderr[-800:]}")
    return json.loads(out.read_text())


def diagnose(d: dict) -> dict:
    s = d["series"]
    t = np.array(s["t"])
    gth = np.array(s["sup_gth"])
    sw = np.array(s["sup_w"])
    th2d = np.array(s["th2_drift"])
    # PHYSICAL trust boundary: theta^2 is a Casimir (exactly conserved in the
    # inviscid limit), so the run is physical only while it holds. This is
    # STRICTER and more meaningful than the tail wire -- the front out-steepens
    # a uniform grid and dumps spurious theta^2 (Gibbs) before the tail band
    # fills. Everything past the first 0.1% Casimir violation is meter garbage.
    trust = th2d < 1e-3
    if not trust.any():
        trust[0] = True
    tt, gg, ww = t[trust], gth[trust], sw[trust]
    out = {"status": d["final"]["status"], "t_trust_end": float(tt[-1]) if len(tt) else 0.0,
           "n_trust": int(trust.sum()), "gth_ratio": float(gg[-1] / gg[0]) if len(gg) > 1 else 1.0,
           "th2_drift": d["final"]["th2_drift"], "itgth": d["final"]["itgth"],
           "sup_w_end": float(ww[-1]) if len(ww) else 0.0}
    if len(tt) >= 8:
        lg = np.log(gg)
        q = len(tt) // 4
        s_last = np.polyfit(tt[-q:], lg[-q:], 1)[0]
        s_prev = np.polyfit(tt[-2 * q:-q], lg[-2 * q:-q], 1)[0]
        out["accel"] = float(s_last / s_prev) if s_prev > 1e-9 else None
        # 1/sup|w| -> 0 extrapolation on the last third (the (T-t)^-1 test)
        m = len(tt) // 3
        inv = 1.0 / ww[-m:]
        A_, r = np.polyfit(tt[-m:], inv, 1, full=True)[:2]
        slope, icept = A_
        out["Tstar"] = float(-icept / slope) if slope < 0 else None
        ss = float(r[0]) if len(r) else 0.0
        out["Tstar_R2"] = float(1 - ss / max(np.sum((inv - inv.mean()) ** 2), 1e-30))
    return out


def main() -> None:
    print("=== bq2 wall geometry: does it blow up? (fp64, convergence-trusted) ===")
    # amplitude scan at a working resolution
    AMPS = [4.0, 8.0, 16.0]
    LADDER = [(192, 144), (256, 192)]
    rows = []
    for A in AMPS:
        d = run_bq2(A, 192, 144, tmax=10.0)
        dg = diagnose(d)
        dg["A"], dg["N"] = A, 192
        rows.append(dg)
        print(f"A={A:>4g} N=192 | {dg['status']:>14} t_trust={dg['t_trust_end']:.2f} "
              f"gth x{dg['gth_ratio']:.1f} | accel={dg.get('accel')} "
              f"Tstar={dg.get('Tstar')} (R2={dg.get('Tstar_R2', 0):.3f}) "
              f"| th2_drift={dg['th2_drift']:.1e}")

    # convergence trust: re-run the most-accelerating A at 256, compare accel
    best = max(rows, key=lambda r: (r.get("accel") or 0))
    d2 = run_bq2(best["A"], 256, 192, tmax=10.0)
    dg2 = diagnose(d2)
    dg2["A"], dg2["N"] = best["A"], 256
    rows.append(dg2)
    print(f"\nconvergence check A={best['A']:g}: "
          f"accel N192={best.get('accel')} vs N256={dg2.get('accel')} | "
          f"Tstar N192={best.get('Tstar')} vs N256={dg2.get('Tstar')} | "
          f"th2_drift {dg2['th2_drift']:.1e}")
    # a real under-resolved singular approach STRENGTHENS with resolution
    # (the theta^2-conserved window reaches closer to T*); a numerical artifact
    # WEAKENS. Compare accel + trusted growth magnitude across the ladder.
    accel_signal = (dg2.get("accel") or 0) > 1.05
    strengthens = ((dg2.get("accel") or 0) >= (best.get("accel") or 0)
                   and dg2["gth_ratio"] >= best["gth_ratio"])
    print(f"\nVERDICT: within the theta^2-conserved window, sup|grad theta| "
          f"{'ACCELERATES' if accel_signal else 'does NOT accelerate'} "
          f"(accel N{best['N']}={best.get('accel'):.2f} -> "
          f"N{dg2['N']}={dg2.get('accel'):.2f}); signal "
          f"{'STRENGTHENS' if strengthens else 'WEAKENS'} with resolution.")
    if accel_signal and strengthens:
        print("-> under-resolved approach to the proven singularity: real "
              "gradient-focusing the parity box lacked, resolution-LIMITED "
              "(front out-steepens uniform N<=256, as Luo-Hou's 2048^2+AMR "
              "predicts). A100 ladder WARRANTED -- to extend the conserved "
              "window and read the (T-t)^-1 / (T-t)^-2 exponents.")
    else:
        print("-> no resolution-strengthening singular signal; gap is IC-"
              "focusing or a real absence, not just resolution.")
    (HERE.parent / "runs" / "bq2_science.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
