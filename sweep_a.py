#!/usr/bin/env python
"""gCLM (a, A) phase-diagram sweep at nu=1, ic=cos2 -- the a-dial campaign.

CLM (a=0) and DG (a=1) endpoints are already measured; this sweeps the
interpolation, dense near the marginal zone a in (0.90, 0.98) found by the
fp64 calibration (lingering blowup T*=7.54 at a=0.95, A=24 -- the first
hover-phenomenology sighting). Each point is a full swarm run with gates,
fp64 anchor, and its own vault note; this driver aggregates A*(a) into a
final phase-curve note.

Run: ~/parzival/.venv/bin/python sweep_a.py
"""
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
PY = str(HERE / ".venv" / "bin" / "python")

# (a, alo, ahi, iters) -- ranges from the fp64 calibration fate maps
POINTS = [
    (0.00, 3.0, 12.0, 8000),
    (0.20, 3.0, 12.0, 8000),
    (0.40, 3.0, 12.0, 8000),
    (0.60, 3.0, 16.0, 8000),
    (0.80, 4.0, 20.0, 8000),
    (0.85, 6.0, 24.0, 8000),
    (0.90, 8.0, 32.0, 8000),
    (0.93, 10.0, 40.0, 15000),
    (0.95, 12.0, 48.0, 15000),
    (0.96, 12.0, 48.0, 15000),
    (0.97, 12.0, 48.0, 15000),
]
BATCH = 16384


def main() -> None:
    t0 = time.time()
    curve = []
    for a, alo, ahi, iters in POINTS:
        out = HERE / "runs" / f"gclm_a{a:g}.json"
        cmd = [PY, str(HERE / "swarm_m1.py"), "--model", "gclm", "--a", str(a),
               "--ic", "cos2", "--alo", str(alo), "--ahi", str(ahi),
               "--batch", str(BATCH), "--iters", str(iters), "--out", str(out)]
        print(f"[sweep] a={a:g} range=[{alo},{ahi}] iters={iters} "
              f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[sweep] a={a:g} FAILED:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}",
                  flush=True)
            continue
        s = json.loads(out.read_text())
        pt = {"a": a,
              "Astar_est": s.get("Astar_est"),
              "Astar_fp64_bracket": s.get("Astar_fp64_bracket"),
              "boundary_cells": s.get("boundary_cells"),
              "resolved": s.get("resolved"), "blowups": s.get("blowups"),
              "hover": s.get("hover_candidates"), "lowtrust": s.get("lowtrust"),
              "dead": s.get("dead_lanes"), "arange": [alo, ahi]}
        curve.append(pt)
        print(f"[sweep] a={a:g} -> A*={pt['Astar_est']} "
              f"fp64={pt['Astar_fp64_bracket']} hover={pt['hover']} "
              f"blow={pt['blowups']}/{pt['resolved']}", flush=True)

    (HERE / "runs" / "gclm_curve.json").write_text(json.dumps(curve, indent=2))

    sys.path.insert(0, str(HERE))
    from swarm_m1 import emit_note
    hover_total = sum(p["hover"] or 0 for p in curve)
    vals = {"nu": 1.0, "ic": "cos2", "batch": BATCH,
            "curve": [{"a": p["a"], "Astar_est": p["Astar_est"],
                       "fp64": p["Astar_fp64_bracket"], "hover": p["hover"]}
                      for p in curve],
            "hover_total": hover_total,
            "endpoints": {"a0_clm_cos": "see swarm-m1-clm notes",
                          "a1_dg": "852k decays, no boundary <=48"}}
    body = ("\ngCLM a-dial phase curve at nu=1 (cos2). Calibration found the "
            "marginal zone a in (0.90, 0.98): lingering blowup T*=7.54 at "
            "a=0.95 A=24 (fp64) -- hover phenomenology one notch below DG. "
            "Full per-a data in runs/gclm_a*.json; each point has its own "
            "chained swarm note.\n")
    h = emit_note(HERE / "vault", "gclm-phase-curve",
                  "gCLM a-sweep nu=1, swarm + fp64 anchors", "quasi", vals,
                  ["critical-threshold", "hover-requires-depletion",
                   "swarm-engine"], body=body)
    print(f"[sweep] done in {(time.time()-t0)/60:.1f} min, "
          f"hover_total={hover_total}, phase-curve note {h}", flush=True)


if __name__ == "__main__":
    main()
