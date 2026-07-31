#!/usr/bin/env python
"""BREAKER attack 3: actual blowup-free dt ceiling vs the engine's choice.

Build the standard smooth state (A=4, N=256, adaptive to t=0.5 -- the gate
4a/4b regime), then integrate a FIXED horizon T=0.3 from that state at fixed
dt, bisecting for the largest dt that survives (finite, theta^2 drift < 1e-2,
sup|w| < 100x). Compare dt_crit against the engine's adaptive choice dt_eng
at the same state, and report which term of the dt law binds.

New file; bq.py untouched. Run:
  /Users/epagogellc/parzival/.venv/bin/python \
      /Users/epagogellc/parzival/boussinesq/breaker_dtceiling.py
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bq import BQ, theta0_hat, DT_MAX, C_ADV, C_BUO  # noqa: E402

N = 256
A = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
T_PREP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
T_TRIAL = 0.3
MIN_STEPS = 40          # never let a trial degenerate to 1-2 steps
DRIFT_TOL = 1e-2
SUPW_FACTOR = 100.0


def prep_state():
    eng = BQ(N)
    w = np.zeros((N, N))
    th = theta0_hat(eng, A)
    t = 0.0
    while t < T_PREP:
        w, th, dt, _ = eng.step(w, th)
        t += dt
    return eng, w, th, t


def trial(eng: BQ, w0, th0, dt: float) -> tuple[bool, str]:
    w, th = w0.copy(), th0.copy()
    q0 = float(np.sum(th ** 2))
    sw0 = float(np.abs(w).max())  # coefficient sup as cheap size proxy
    steps = max(int(math.ceil(T_TRIAL / dt)), MIN_STEPS)
    for _ in range(steps):
        w, th, _, _ = eng.step(w, th, dt=dt)
        if not (np.isfinite(w).all() and np.isfinite(th).all()):
            return False, "nonfinite"
        if float(np.abs(w).max()) > SUPW_FACTOR * max(sw0, 1.0):
            return False, "sup_w blowup"
    drift = abs(float(np.sum(th ** 2)) - q0) / q0
    if drift > DRIFT_TOL:
        return False, f"theta^2 drift {drift:.2e}"
    return True, f"stable (theta^2 drift {drift:.2e}, {steps} steps)"


def main() -> None:
    eng, w, th, t0 = prep_state()
    _, _, aux = eng.rhs(w, th)
    dt_eng = eng.dt_of(aux)
    dt_buo = C_BUO / math.sqrt(aux["sup_gth"] + 1e-300)
    dt_adv = C_ADV / (eng.K * aux["sup_u"]) if aux["sup_u"] > 0 else math.inf
    binder = min((DT_MAX, "DT_MAX"), (dt_buo, "buoyancy"), (dt_adv, "advective"))
    print(f"state at t={t0:.4f}: sup_u {aux['sup_u']:.4f} "
          f"sup_gth {aux['sup_gth']:.4f}")
    print(f"dt law terms: DT_MAX {DT_MAX:g} | buoyancy {dt_buo:.4e} | "
          f"advective {dt_adv:.4e} -> engine dt {dt_eng:.4e} "
          f"(binding: {binder[1]})")

    # bracket upward from dt_eng
    lo, hi = dt_eng, None
    d = dt_eng
    while hi is None:
        d *= 2.0
        ok, why = trial(eng, w, th, d)
        print(f"  probe dt={d:.4e}: {why}")
        if ok:
            lo = d
        else:
            hi = d
        if d > 1.0:
            print("no ceiling found below dt=1; abort")
            return
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        ok, why = trial(eng, w, th, mid)
        print(f"  bisect dt={mid:.4e}: {why}")
        lo, hi = (mid, hi) if ok else (lo, mid)
    dt_crit = lo
    print(f"\nengine adaptive dt at state : {dt_eng:.4e}")
    print(f"blowup-free ceiling dt_crit : {dt_crit:.4e}  (hi bound {hi:.4e})")
    print(f"safety margin dt_crit/dt_eng: {dt_crit / dt_eng:.2f}x "
          f"(binding constraint at this state: {binder[1]})")


if __name__ == "__main__":
    main()
