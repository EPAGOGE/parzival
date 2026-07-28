#!/usr/bin/env python
"""BREAKER attack 1: long-integration equivariance drift under ADAPTIVE dt.

Gate 5 proves S1/S2 equivariance over 1000 FIXED-dt steps on a gentle state
(A=4). This attack integrates a hot state (A=6, N=128) to t >= 4 with the
engine's own adaptive dt (dt=None in step) and measures the violation of the
same two discrete involutions over thousands of steps, checkpointing along
the way. It also tracks whether the three adaptive dt sequences (base, S1, S2)
ever diverge bitwise -- adaptive dt is a function of sup-norms, which are
invariant under the involutions only if the fp reductions land identically.

New file; bq.py untouched. Run:
  /Users/epagogellc/parzival/.venv/bin/python \
      /Users/epagogellc/parzival/boussinesq/breaker_equivariance_long.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bq import BQ, theta0_hat  # noqa: E402

N = 128
A = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
TMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
MAX_STEPS = 200_000
CHECK_EVERY = 250

sgn_s = np.where(np.arange(1, N + 1) % 2 == 1, -1.0, 1.0)
sgn_c = np.where(np.arange(0, N) % 2 == 1, -1.0, 1.0)


def s1(w, th):
    return w * sgn_s[:, None], th * sgn_c[:, None]


def s2(w, th):
    return w * sgn_s[None, :], th * sgn_s[None, :]


def main() -> None:
    eng = BQ(N)
    w0 = np.zeros((N, N))
    th0 = theta0_hat(eng, A)

    # three trajectories, each with its OWN adaptive dt
    traj = {
        "base": [w0.copy(), th0.copy()],
        "S1": list(s1(w0.copy(), th0.copy())),
        "S2": list(s2(w0.copy(), th0.copy())),
    }
    t = {k: 0.0 for k in traj}
    dt_diverged_at = {"S1": None, "S2": None}

    rows = []
    step = 0
    while t["base"] < TMAX and step < MAX_STEPS:
        dts = {}
        for k, (w, th) in traj.items():
            w2, th2, dt, _ = eng.step(w, th)          # adaptive: dt=None
            if not (np.isfinite(w2).all() and np.isfinite(th2).all()):
                print(f"NONFINITE in {k} at step {step}, t={t[k]:.4f}")
                report(rows, dt_diverged_at, t, step, eng, traj)
                return
            traj[k][0], traj[k][1] = w2, th2
            t[k] += dt
            dts[k] = dt
        for k in ("S1", "S2"):
            if dts[k] != dts["base"] and dt_diverged_at[k] is None:
                dt_diverged_at[k] = (step, dts["base"], dts[k])
        step += 1
        if step % CHECK_EVERY == 0 or t["base"] >= TMAX:
            fw, ft = traj["base"]
            row = {"step": step, "t": t["base"], "dt": dts["base"]}
            for k, S in (("S1", s1), ("S2", s2)):
                sw, st = S(fw, ft)
                ew = float(np.abs(sw - traj[k][0]).max()) / float(np.abs(fw).max())
                et = float(np.abs(st - traj[k][1]).max()) / float(np.abs(ft).max())
                row[f"{k}_w_rel"] = ew
                row[f"{k}_th_rel"] = et
                row[f"{k}_bitexact"] = bool(
                    np.array_equal(sw, traj[k][0]) and np.array_equal(st, traj[k][1]))
            b = eng.budgets(fw, ft)
            row["tail_w"] = b["tail_w"]
            row["tail_th"] = b["tail_th"]
            rows.append(row)
            print(f"step {step:6d} t={row['t']:.4f} dt={row['dt']:.3e} "
                  f"S1 w/th {row['S1_w_rel']:.3e}/{row['S1_th_rel']:.3e} "
                  f"(bit={row['S1_bitexact']}) "
                  f"S2 w/th {row['S2_w_rel']:.3e}/{row['S2_th_rel']:.3e} "
                  f"(bit={row['S2_bitexact']}) "
                  f"tails {row['tail_w']:.2e}/{row['tail_th']:.2e}", flush=True)
    report(rows, dt_diverged_at, t, step, eng, traj)


def report(rows, dt_diverged_at, t, step, eng, traj) -> None:
    out = {"N": N, "A": A, "steps": step, "t_final": t["base"],
           "dt_diverged_at": {k: (None if v is None else
                                  {"step": v[0], "dt_base": v[1], "dt_S": v[2]})
                              for k, v in dt_diverged_at.items()},
           "rows": rows}
    p = pathlib.Path(__file__).resolve().parent / "runs" / "breaker_equiv_long.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"\nsteps={step} t_final={t['base']:.4f}")
    print(f"dt sequence divergence: {out['dt_diverged_at']}")
    if rows:
        worst = max(max(r["S1_w_rel"], r["S1_th_rel"],
                        r["S2_w_rel"], r["S2_th_rel"]) for r in rows)
        allbit = all(r["S1_bitexact"] and r["S2_bitexact"] for r in rows)
        print(f"worst equivariance violation over run: {worst:.3e}")
        print(f"bit-exact at every checkpoint: {allbit}")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
