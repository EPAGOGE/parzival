#!/usr/bin/env python
"""Amplitude dial on the Boussinesq corner scenario (era B1, first science).

Inviscid regime: there is no decay fate. The measurables per A are growth
CHARACTER of sup|grad theta| -- how fast, whether accelerating, and where the
meter escapes (tail_exhausted/dt_exhausted are honest resolution-limit
statuses, never blowup claims). N=256 pass; the interesting band re-runs at
N=512 next (resolution ladder).

Run: ~/parzival/.venv/bin/python boussinesq/bq_dial.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
PY = str(HERE.parent / ".venv" / "bin" / "python")
AMPS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 16.0]
N, TMAX = 256, 8.0


def growth_character(d: dict) -> dict:
    t = np.array(d["series"]["t"])
    g = np.array(d["series"]["sup_gth"])
    ok = g > 0
    t, g = t[ok], g[ok]
    out = {"g0": float(g[0]), "g_end": float(g[-1]),
           "ratio_end": float(g[-1] / g[0]), "t_end": float(t[-1])}
    for r, key in ((10.0, "t_10x"), (100.0, "t_100x")):
        idx = np.argmax(g >= r * g[0])
        out[key] = float(t[idx]) if g[idx] >= r * g[0] else None
    if len(t) >= 8:
        lg = np.log(g)
        q = len(t) // 4
        s_last = float(np.polyfit(t[-q:], lg[-q:], 1)[0]) if q >= 2 else None
        s_prev = float(np.polyfit(t[-2 * q:-q], lg[-2 * q:-q], 1)[0]) \
            if q >= 2 else None
        out["logslope_last"] = s_last
        out["accel"] = (s_last / s_prev) if (s_last and s_prev and s_prev > 0) \
            else None
    return out


def main() -> None:
    t0 = time.time()
    rows = []
    for A in AMPS:
        out = HERE.parent / "runs" / f"bsq_dial_A{A:g}.json"
        cmd = [PY, str(HERE / "bq.py"), "--scenario", "--A", str(A),
               "--N", str(N), "--tmax", str(TMAX), "--out", str(out)]
        print(f"[dial] A={A:g} ...", flush=True)
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[dial] A={A:g} FAILED\n{r.stdout[-800:]}\n{r.stderr[-500:]}",
                  flush=True)
            continue
        d = json.loads(out.read_text())
        fin = d["final"]
        row = {"A": A, "status": fin["status"], "t_end": fin["t"],
               "low_trust_since_t": fin.get("low_trust_since_t"),
               "sup_w_end": fin["sup_w"], "tail_w_end": fin["tail_w"],
               "bkm_I": fin.get("bkm_I"), **growth_character(d)}
        rows.append(row)
        print(f"[dial] A={A:g} -> {fin['status']} t_end={fin['t']:.3f} "
              f"grad-ratio={row['ratio_end']:.1f} t_10x={row['t_10x']} "
              f"accel={row.get('accel')}", flush=True)
    (HERE.parent / "runs" / "bsq_dial.json").write_text(
        json.dumps({"N": N, "tmax": TMAX, "rows": rows}, indent=2))
    print(f"[dial] done in {(time.time()-t0)/60:.1f} min "
          f"({len(rows)}/{len(AMPS)} runs)", flush=True)


if __name__ == "__main__":
    main()
