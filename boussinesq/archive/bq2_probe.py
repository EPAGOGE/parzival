#!/usr/bin/env python
"""Conserved-window probe: integrate bq2 to the theta^2 break, no further.

The full bq2 scenario death-grinds past the Casimir break (front out-steepens
the grid, dt -> DT_MIN, all wasted since we discard that regime). This drives
the SAME validated BQ2 engine but stops as soon as theta^2 drifts past the
trust boundary -- exactly the window we diagnose -- with live progress. Gives
the N-ladder point fast and visibly.

Run: bq2_probe.py <nx> <ny>   (e.g. 384 288)
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from bq2 import BQ2, theta0

TRUST = 1e-3          # theta^2 Casimir trust boundary (diagnose window)
STOP = 3e-3           # integrate a hair past it, then stop (skip the grind)


def main() -> None:
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 384
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 288
    A = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    eng = BQ2(nx, ny)
    th = theta0(eng, A)
    w = np.zeros_like(th)
    th2_0 = eng.integ(th ** 2)
    t, si, t0 = 0.0, 0, time.time()
    ser = {"t": [], "sup_w": [], "sup_gth": [], "th2_drift": []}
    while si < 200000:
        w, th, dt, aux = eng.step(w, th)
        t += dt
        si += 1
        if si % 25 == 0 or si == 1:
            d = abs(eng.integ(th ** 2) - th2_0) / max(th2_0, 1e-300)
            ser["t"].append(t)
            ser["sup_w"].append(float(np.abs(w).max()))
            ser["sup_gth"].append(aux["sup_gth"])
            ser["th2_drift"].append(d)
            if si % 500 == 0:
                print(f"  t={t:.4f} step={si} sup|gth|={aux['sup_gth']:.3e} "
                      f"th2_drift={d:.2e} dt={dt:.1e} ({time.time()-t0:.0f}s)",
                      flush=True)
            if d > STOP:
                print(f"  theta^2 break at t={t:.4f} (drift {d:.2e}) -- stop",
                      flush=True)
                break
        if dt < 1e-9:
            print(f"  dt exhausted at t={t:.4f}", flush=True)
            break

    tt = np.array(ser["t"])
    gg = np.array(ser["sup_gth"])
    ww = np.array(ser["sup_w"])
    dd = np.array(ser["th2_drift"])
    trust = dd < TRUST
    tb = float(tt[trust][-1]) if trust.any() else 0.0
    out = {"nx": nx, "ny": ny, "A": A, "t_trust_end": tb,
           "n_trust": int(trust.sum())}
    if trust.sum() >= 6:
        tg, ggt = tt[trust], gg[trust]
        q = len(tg) // 3
        s_last = np.polyfit(tg[-q:], np.log(ggt[-q:]), 1)[0]
        s_prev = np.polyfit(tg[-2 * q:-q], np.log(ggt[-2 * q:-q]), 1)[0]
        out["gth_ratio"] = float(ggt[-1] / ggt[0])
        out["accel"] = float(s_last / s_prev)
        out["supw_trust_end"] = float(ww[trust][-1])
    pathlib.Path(f"../runs/bq2probe_N{nx}.json").write_text(json.dumps(out, indent=2))
    print(f"N={nx}x{ny} A={A:g}: window t<={tb:.3f} | gth x{out.get('gth_ratio',0):.1f} "
          f"| accel={out.get('accel',0):.2f} | sup|w|_end={out.get('supw_trust_end',0):.2f} "
          f"| {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
