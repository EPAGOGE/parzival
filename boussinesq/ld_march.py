#!/usr/bin/env python3
"""Large-deviation march (pre-registered: NU_CURVE_SPEC.md). Reuses
m2_ampladder.march_one_amp verbatim (its constants: 240 steps, ds=0.25,
seed 0, M2 stop rule)."""
import pathlib
import time

import numpy as np

from m2_ampladder import DS, march_one_amp, s_transient_of
from march_s import QuasiNewtonSMarcher, _load_production_pnorm_certificate

AMPS = [3e-3, 1e-2, 3e-2, 1e-1]
V_FLOOR = 5e-16
OUT = pathlib.Path(__file__).parent / "LD_MARCH.out"
_lines = []


def log(s=""):
    _lines.append(s)
    print(s, flush=True)


def main():
    t0 = time.time()
    log("=" * 78)
    log("LARGE-DEVIATION MARCH -- does the s=0.5 transient survive?")
    log("=" * 78)
    cert = _load_production_pnorm_certificate()
    log(f"cert: root fresh ||F||max={cert['root_fmax_check']:.2e}  "
        f"cholesky {'PASS' if cert['cholesky_ok'] else 'FAIL'}")
    if not cert["root_ok"] or not cert["cholesky_ok"]:
        log("ABORT -- certificate did not re-verify.")
        OUT.write_text("\n".join(_lines) + "\n")
        return
    M = QuasiNewtonSMarcher(cert["S"], cert["R0"], cert["Lred"],
                            cert["Z0"], sign=+1.0)
    M.frozen_lu(DS)
    ok_bar = True
    for amp in AMPS:
        log("-" * 78)
        log(f"AMP {amp:g}")
        res = march_one_amp(cert, M, amp)
        raws, Vs = res["raws"], res["Vs"]
        n = res["n_done"]
        ipk = int(np.argmax(raws))
        s_pk = ipk * DS
        pk_x = raws[ipk] / max(raws[0], 1e-300)
        _sp, _px, s_tr = s_transient_of(raws)
        genuine = [(i, Vs[i], (Vs[i] - Vs[i - 1]) / abs(Vs[i - 1]))
                   for i, _p, _v, g in
                   [(v[0], v[1], v[2], v[3]) for v in res["violations"]]
                   if Vs[i] > 100 * V_FLOOR] if res["violations"] else []
        log(f"  steps {n}/240  stop={res['stop']['kind'] if res['stop'] else 'none'}"
            f"  s_peak={s_pk:g}  peak={pk_x:.4f}x  "
            f"s_transient={s_tr if s_tr is not None else 'not reached'}")
        log(f"  V: {Vs[0]:.3e} -> {Vs[-1]:.3e}   raw: {raws[0]:.3e} -> "
            f"{raws[-1]:.3e}")
        if res["violations"]:
            gen = [v for v in res["violations"] if Vs[v[0]] > 100 * V_FLOOR]
            if gen:
                first = gen[0]
                log(f"  GENUINE P-NORM GROWTH: {len(gen)} violations above "
                    f"100x floor, first at step {first[0]} "
                    f"(s={first[0] * DS:g}, V={Vs[first[0]]:.2e}, "
                    f"rel_dV={first[3]:.2e}) -- nonlinear signature, "
                    f"march residual-valid at those steps")
            else:
                log(f"  {len(res['violations'])} V-violations, ALL within "
                    f"100x of the 5e-16 floor -- artifacts (M2 adjudication)")
        if n * DS >= s_pk + DS and not (0.25 <= s_pk <= 0.75):
            ok_bar = False
            log(f"  BAR MISS: s_peak={s_pk:g} outside [0.25, 0.75]")
    log("")
    log("=" * 78)
    log(f"VERDICT (pre-registered bar): transient timescale "
        f"{'ROBUST -- s_peak in [0.25,0.75] at every valid amp' if ok_bar else 'NOT robust -- see BAR MISS lines'}")
    log(f"total wall: {time.time() - t0:.1f}s")
    OUT.write_text("\n".join(_lines) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
