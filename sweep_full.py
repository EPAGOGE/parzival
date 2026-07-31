#!/usr/bin/env python
"""The full simulation: complete (a, A) phase portrait of viscous gCLM at nu=1
(nu is exact gauge -- rung1-edge-branch), all instruments live.

Stage A: fp64 edge-branch continuation over the full a-grid -> Omega(a),
         lam_u(a), drift c(a). Omega sets each swarm window automatically.
Stage B: 24 swarm points (B=32768), dense near the pole AND in the small-a
         branch-exchange region; per-cell ledger serialized for the
         hover-rate surface.
Aggregates runs/full_map.json and vaults the portrait.

Run: ~/parzival/.venv/bin/python sweep_full.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
PY = str(HERE / ".venv" / "bin" / "python")

from rung1b_validate import make_ops, newton  # noqa: E402  (NU=1 inside)
from swarm_m1 import macro_step_np, make_ic   # noqa: E402

A_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7,
          0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.93, 0.94, 0.95, 0.96, 0.965, 0.97]
BATCH = 32768
SEED_A, SEED_AMP = 0.93, 18.13614   # threshold bisected in rung1


def stage_a() -> dict[float, dict]:
    o = make_ops(128)
    mats = {k: o[k] for k in "HLDG"}
    w = make_ic(np.array([SEED_AMP]), 128, "cos2")
    t = np.zeros(1)
    best, br, sp, tp = None, np.inf, SEED_AMP, 0.0
    for _ in range(150000):
        w, t = macro_step_np(w, t, mats, 1.0, "gclm", a=SEED_A)
        s, tn = float(np.max(np.abs(w))), float(t[0])
        if s > 1e3 or (s < 0.1 * SEED_AMP and tn > 0.5):
            break
        if tn > 1.0 and tn - tp > 1e-6:
            r = abs(np.log(s) - np.log(sp)) / (tn - tp)
            if r < br:
                br, best = r, w[0].copy()
        sp, tp = s, tn
    ws, _, cs = newton(best, SEED_A, best, o)
    print(f"[full] stage A seed: sup={np.max(np.abs(ws)):.4f}", flush=True)

    def lam_u(wv, a, c):
        from rung1b_validate import jac
        ev = np.linalg.eigvals(jac(wv, a, o) + c * o["D"])
        return float(np.sort(ev.real)[::-1][0])

    edge = {SEED_A: {"omega": float(np.max(np.abs(ws))), "c": cs,
                     "lam_u": lam_u(ws, SEED_A, cs)}}
    for targets in ([a for a in sorted(A_GRID) if a > SEED_A],
                    [a for a in sorted(A_GRID, reverse=True) if a < SEED_A]):
        wc, cc = ws.copy(), cs
        prev = SEED_A
        for at in targets:
            ok = True
            for sub in np.linspace(prev, at, max(2, int(abs(at - prev) / 0.05) + 2))[1:]:
                wc, _, cc = newton(wc, float(sub), wc, o, c0=cc)
                if wc is None:
                    print(f"[full] stage A stopped near a={sub:.3f}", flush=True)
                    ok = False
                    break
            if not ok:
                break
            edge[round(at, 4)] = {"omega": float(np.max(np.abs(wc))), "c": cc,
                                  "lam_u": lam_u(wc, float(at), cc)}
            prev = at
    for a in sorted(edge):
        e = edge[a]
        print(f"[full] edge a={a:.3f} Omega={e['omega']:8.3f} "
              f"lam_u={e['lam_u']:7.3f} c={e['c']:8.4f}", flush=True)
    return edge


def swarm_window(a: float, omega: float | None) -> tuple[float, float]:
    if a <= 0.45 or omega is None:
        return 3.0, 12.0          # branch is not the edge here; fate data rules
    return (max(3.0, round(0.45 * omega, 1)),
            min(58.0, round(2.2 * omega, 1)))


def main() -> None:
    t0 = time.time()
    edge = stage_a()
    rows = []
    for a in A_GRID:
        om = edge.get(round(a, 4), {}).get("omega")
        alo, ahi = swarm_window(a, om)
        iters = 15000 if a >= 0.92 else 8000
        out = HERE / "runs" / f"full_a{a:g}.json"
        cmd = [PY, str(HERE / "swarm_m1.py"), "--model", "gclm", "--a", str(a),
               "--ic", "cos2", "--alo", str(alo), "--ahi", str(ahi),
               "--batch", str(BATCH), "--iters", str(iters), "--out", str(out)]
        print(f"[full] a={a:g} window=[{alo},{ahi}] iters={iters} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[full] a={a:g} FAILED\n{r.stdout[-1500:]}\n{r.stderr[-800:]}",
                  flush=True)
            continue
        s = json.loads(out.read_text())
        e = edge.get(round(a, 4), {})
        rows.append({"a": a, "Astar_est": s.get("Astar_est"),
                     "Astar_fp64_bracket": s.get("Astar_fp64_bracket"),
                     "hover": s.get("hover_candidates"),
                     "resolved": s.get("resolved"), "blowups": s.get("blowups"),
                     "lowtrust": s.get("lowtrust"), "dead": s.get("dead_lanes"),
                     "arange": [alo, ahi], "iters": iters,
                     "omega_edge": e.get("omega"), "lam_u": e.get("lam_u"),
                     "c_drift": e.get("c"), "cells": s.get("cells")})
        print(f"[full] a={a:g} -> A*={s.get('Astar_est')} "
              f"fp64={s.get('Astar_fp64_bracket')} Omega={e.get('omega')} "
              f"hover={s.get('hover_candidates')}", flush=True)

    (HERE / "runs" / "full_map.json").write_text(json.dumps(
        {"nu": 1.0, "batch": BATCH, "edge": edge, "rows": rows}, indent=2))

    from swarm_m1 import emit_note
    vals = {"points": len(rows), "batch": BATCH,
            "resolved_total": int(sum(r["resolved"] or 0 for r in rows)),
            "hover_total": int(sum(r["hover"] or 0 for r in rows)),
            "curve": [{"a": r["a"], "Astar": r["Astar_est"],
                       "Omega_edge": r["omega_edge"]} for r in rows]}
    body = ("\nFull (a, A) phase portrait at nu=1 (nu exact gauge). Swarm fate "
            "boundary, fp64 anchors, and the edge-state branch Omega(a) "
            "measured together on one grid; per-cell ledgers serialized for "
            "the hover-rate surface. Windows auto-set from Omega(a) for "
            "a>0.45. Full data: runs/full_map.json + runs/full_a*.json.\n")
    h = emit_note(HERE / "vault", "full-phase-portrait",
                  "gCLM full simulation, swarm + edge branch, N=128", "quasi",
                  vals, ["critical-threshold", "hover-requires-depletion",
                         "swarm-engine", "noise-blur"], body=body)
    print(f"[full] done in {(time.time()-t0)/60:.1f} min, note {h}", flush=True)


if __name__ == "__main__":
    main()
