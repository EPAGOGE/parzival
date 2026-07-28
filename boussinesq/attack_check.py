#!/usr/bin/env python
"""Adversarial diagnostic audit for bq.py scenario JSON (breaker pass, fate lens).

Reads runs/atk_*.json produced by:
  bq.py --scenario --A 0.05 --N 128 --tmax 6 --out runs/atk_smallA.json
  bq.py --scenario --A 4    --N 128 --tmax 2 --out runs/atk_A4_N128.json
  bq.py --scenario --A 4    --N 256 --tmax 2 --out runs/atk_A4_N256.json
  bq.py --scenario --A 4    --N 64  --tmax 2 --diag-every 1000000 --out runs/atk_bypass.json

Checks: series integrity (lengths, t monotone, dt vs CFL law, row spacing),
final-dict consistency, bounded-run false-alarm scan, D7 margin sign,
rmax argmax artifact, and N=128-vs-256 per-diagnostic resolution stability
on a common interpolated time grid. Read-only: touches no engine file.
"""
import json
import math
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
DT_MAX, C_BUO = 5e-3, 0.5
SERIES_KEYS = ("t", "dt", "sup_w", "sup_gth", "E", "P", "Z", "th2", "prod",
               "buoy_work", "ep_drift", "th2_drift", "tail_w", "tail_th",
               "rmax", "bkm_I", "d7_margin", "th_overshoot")


def load(name):
    return json.load(open(HERE / "runs" / name))


def integrity(name, d):
    s, f, hdr = d["series"], d["final"], d["header"]
    print(f"\n=== integrity: {name} (N={hdr['N']}, A={hdr['A']}, "
          f"diag_every={hdr['diag_every']}) ===")
    lens = {k: len(s[k]) for k in SERIES_KEYS}
    ok_len = len(set(lens.values())) == 1
    print(f"series lengths equal: {ok_len} ({lens['t']} rows)")
    t = np.array(s["t"])
    dt = np.array(s["dt"])
    mono = bool(np.all(np.diff(t) > 0))
    print(f"t strictly monotone: {mono}")
    # final dict vs last row
    for k in ("sup_w", "sup_gth", "ep_drift", "th2_drift", "tail_w"):
        match = f[k] == s[k][-1]
        if not match:
            print(f"  MISMATCH final[{k}]={f[k]} vs last row {s[k][-1]}")
    print(f"final t == last row t: {f['t'] == s['t'][-1]}")
    # dt law: dt <= min(DT_MAX, C_BUO/sqrt(sup_gth)) rowwise (sup_u unrecorded
    # so the advective limb is unverifiable from the JSON alone)
    gth = np.array(s["sup_gth"])
    cap = np.minimum(DT_MAX, C_BUO / np.sqrt(gth + 1e-300))
    bad = np.where(dt > cap * (1 + 1e-12))[0]
    print(f"rows with dt > buoyancy/DT_MAX cap: {list(bad)}"
          + (f"  [dt={dt[bad]} cap={cap[bad]}]" if len(bad) else ""))
    # row spacing vs diag_every * dt (final row is a partial window by design)
    de = hdr["diag_every"]
    if len(t) > 2:
        ratio = np.diff(t)[:-1] / (de * dt[:-2])
        print(f"interior row spacing / (diag_every*dt): "
              f"min {ratio.min():.3f} max {ratio.max():.3f}")
        print(f"final-row spacing / (diag_every*dt): "
              f"{(t[-1]-t[-2]) / (de * dt[-2]):.4f} (partial window expected)")
    print(f"status={f['status']} low_trust_since_t={f['low_trust_since_t']} "
          f"final tail_w={f['tail_w']:.3e} tail_th={s['tail_th'][-1]:.3e}")
    print(f"d7_margin min over run: {min(s['d7_margin']):.6e} "
          f"(negative = D7 tripwire) | final d7_violation={f['d7_violation']:.3e}")
    print(f"rmax[0]={s['rmax'][0]:.4f} with sup_w[0]={s['sup_w'][0]} "
          f"<-- argmax-of-zeros artifact if sup_w==0")
    print(f"th_overshoot max: {max(s['th_overshoot']):.3e}")


def bounded_scan(d):
    s, f = d["series"], d["final"]
    print("\n=== bounded-run (A=0.05, t=6) false-alarm scan ===")
    print(f"status: {f['status']} (want completed)")
    print(f"low_trust_since_t: {f['low_trust_since_t']} (want None)")
    print(f"max th2_drift: {max(s['th2_drift']):.3e}")
    print(f"max ep_drift:  {max(s['ep_drift']):.3e}")
    print(f"max tail_w:    {max(s['tail_w']):.3e}  max tail_th: {max(s['tail_th']):.3e}")
    print(f"max th_overshoot: {max(s['th_overshoot']):.3e} "
          f"(vs sup_theta0 {d['header']['sup_theta0_grid']:.3f})")
    print(f"min d7_margin: {min(s['d7_margin']):.6e} (want >= 0)")
    print(f"sup_w range: [{min(s['sup_w']):.4f}, {max(s['sup_w']):.4f}]")
    print(f"bkm_I final: {f['bkm_I']:.4f} (linear-ish growth expected, "
          f"rate ~ {f['bkm_I']/f['t']:.4f}/unit t)")
    # is dt pinned at DT_MAX (quiet state) the whole way?
    dt = np.array(s["dt"])
    print(f"dt: min {dt.min():.3e} max {dt.max():.3e} "
          f"(DT_MAX={DT_MAX}; pinned={bool(np.all(dt == DT_MAX))})")


def compare(d1, d2):
    s1, s2 = d1["series"], d2["series"]
    n1, n2 = d1["header"]["N"], d2["header"]["N"]
    t1, t2 = np.array(s1["t"]), np.array(s2["t"])
    tmax = min(t1[-1], t2[-1])
    # common grid strictly inside both series; skip t=0 rows (zero denominators)
    tg = np.linspace(0.05, tmax * 0.999, 200)
    print(f"\n=== N={n1} vs N={n2} on common t-grid [0.05, {tmax:.3f}] ===")
    print(f"{'diag':>12} {'max rel diff':>13} {'rel diff @t=1.0':>16} "
          f"{'rel diff @t=' + format(tmax*0.999, '.2f'):>16}")
    for k in ("sup_w", "sup_gth", "E", "P", "Z", "th2", "prod", "buoy_work",
              "bkm_I", "d7_margin", "tail_w", "tail_th", "rmax", "dt",
              "th_overshoot", "ep_drift", "th2_drift"):
        v1 = np.interp(tg, t1, np.array(s1[k]))
        v2 = np.interp(tg, t2, np.array(s2[k]))
        scale = np.maximum(np.abs(v2), 1e-300)
        rel = np.abs(v1 - v2) / scale
        i10 = np.argmin(np.abs(tg - 1.0))
        print(f"{k:>12} {rel.max():>13.3e} {rel[i10]:>16.3e} {rel[-1]:>16.3e}")
    # where do the two runs' argmax locations sit late in the run?
    for s, n in ((s1, n1), (s2, n2)):
        i = np.searchsorted(np.array(s["t"]), tmax) - 1
        print(f"N={n}: at t={s['t'][i]:.3f} rmax={s['rmax'][i]:.4f} "
              f"sup_w={s['sup_w'][i]:.3f}")


def main():
    runs = {n: load(n) for n in ("atk_smallA.json", "atk_A4_N128.json",
                                 "atk_A4_N256.json", "atk_bypass.json")}
    for n, d in runs.items():
        integrity(n, d)
    bounded_scan(runs["atk_smallA.json"])
    compare(runs["atk_A4_N128.json"], runs["atk_A4_N256.json"])


if __name__ == "__main__":
    main()
