#!/usr/bin/env python
"""Tier-Q shadow auditor: redundancy-based SDC detection for the swarm.

The in-loop M1/M2 monitors (audit.py, swarm_m1.macro_step audit=True) are
CONSISTENCY checks -- they compare a state against an invariant it must
satisfy. They catch gross corruption (NaN/Inf, broadband state damage,
operator faults) but are STRUCTURALLY BLIND to a consistent-ALU SDC: a bad
multiplier that corrupts a stage flows equally into the update and its own
consistency check, so they move together and the residual stays small
(verified 2026-07-23: a 2x k3 fault read m1=0.09, under the healthy tail).

Catching that requires REDUNDANCY, not consistency: recompute the same lanes
by an INDEPENDENT path (CPU fp64, gate-validated implementation-exact to the
GPU fp32 kernel) and compare trajectories. A GPU-side SDC corrupts the GPU
trajectory; the fp64 recompute follows truth; they diverge -> caught.

The design subtlety this module exists to handle: fp32 vs fp64 LEGITIMATELY
diverges (measured ~7% near threshold -- the known fp32 characteristic, not
corruption). So the verdict is never "fp32 != fp64". It is:
  (1) cohort-outlier: a lane diverging far more than same-regime peers
      (robust median+MAD within amplitude bins) -> rare per-lane SDC;
  (2) systematic: a whole cohort's median divergence exceeding the known
      fp32 envelope -> a card-wide fault the outlier test would miss.

This is a periodic biopsy, decoupled from the main loop's recycling: sample
live lanes, cast their fp32 state to fp64, march BOTH forward W steps under a
SHARED dt schedule (so they stay at matched t and divergence is pure state
error), score, report. Never gates fate -- flags + downgrades trust, and the
main loop recycles flagged lanes. Probe-is-not-the-loss holds.
"""
from __future__ import annotations

import numpy as np

from swarm_m1 import DT_CFL, DT_MAX, rhs_np

# Healthy fp32-vs-fp64 divergence over a W=100 biopsy from a shared seed,
# MEASURED on clean gclm lanes 2026-07-23: cohort-median 4.9e-7, p99 1.5e-4,
# max 2.7e-5. (Both paths start from the identical state, so short-window
# divergence is small -- the ~7% fp32 figure is a full-trajectory-to-blowup
# number, not a windowed one.) Floor set ~7x above clean p99.
FP32_ENVELOPE = 1e-3    # cohort-median divergence above this = systematic fault
                        # (clean cohort-median 6e-7, max 6e-5; a 2% card-wide
                        # fault reads 2.7e-3 -> fires with 3x margin)
K_MAD = 12.0            # per-lane outlier bar: median + K_MAD * MAD
DIV_FLOOR = 1e-3        # absolute floor a lane must clear to flag -- keeps a
                        # uniformly-at-roundoff cohort (MAD ~ 0) from flagging.
# SENSITIVITY, stated plainly: detects an SDC whose windowed trajectory
# divergence clears ~1e-3. That scales as (fault size) x (window amplification).
# So it is SHARP in the amplifying near-threshold regime -- exactly where an
# SDC could flip a fate and bias A* -- and blind to small faults in contracting
# regimes, where the SDC is fate-irrelevant. The residual (subtle faults that
# still flip a marginal fate) is covered by the fate-level re-resolution audit.


def _rk4_np(w, dt, mats, nu, model, a):
    k1 = rhs_np(w, mats, nu, model, a)
    k2 = rhs_np(w + 0.5 * dt * k1, mats, nu, model, a)
    k3 = rhs_np(w + 0.5 * dt * k2, mats, nu, model, a)
    k4 = rhs_np(w + dt * k3, mats, nu, model, a)
    return w + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def shadow_audit(engine, w_sample, a0_sample, mats, model, a,
                 window: int = 100) -> dict:
    """Biopsy a set of live lanes. w_sample: (S, N) fp32 torch states pulled
    from the main batch. a0_sample: their launch amplitudes (regime proxy for
    cohorting). Returns per-lane divergence, corruption flags, and verdict."""
    tc = engine.torch
    S = w_sample.shape[0]
    # identical fp64 seed on both paths: the exact fp32 values, widened.
    # cast on CPU -- MPS has no fp64.
    w_cpu = w_sample.cpu().numpy().astype(np.float64)
    w_gpu = w_sample.clone()
    nu = engine.nu
    div = np.zeros(S)                       # running max relative divergence
    for _ in range(window):
        # shared dt from the TRUSTWORTHY fp64 path (batch-max, single schedule
        # so both integrate identical t; divergence is state error not dt drift)
        m64 = np.max(np.abs(w_cpu), axis=1)
        dt = float(min(DT_MAX, DT_CFL / max(1.0, float(m64.max()))))
        w_cpu = _rk4_np(w_cpu, dt, mats, nu, model, a)
        # GPU path steps with the SAME imposed dt (isolated small batch)
        dtt = tc.full((S, 1), dt, dtype=w_gpu.dtype, device=engine.dev)
        k1 = engine.rhs(w_gpu)
        k2 = engine.rhs(w_gpu + 0.5 * dtt * k1)
        k3 = engine.rhs(w_gpu + 0.5 * dtt * k2)
        k4 = engine.rhs(w_gpu + dtt * k3)
        w_gpu = w_gpu + (dtt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        wg = w_gpu.cpu().numpy().astype(np.float64)
        scale = np.maximum(np.max(np.abs(w_cpu), axis=1), 1.0)
        step_div = np.max(np.abs(wg - w_cpu), axis=1) / scale
        step_div = np.where(np.isfinite(step_div), step_div, np.inf)
        div = np.maximum(div, step_div)
        if not np.isfinite(w_cpu).all():    # reference blew up: window done
            break

    # cohort-relative outlier test: bin by launch amplitude (regime), robust
    # median + MAD within each bin; flag lanes far above their peers.
    flags = np.zeros(S, bool)
    nbins = max(1, min(8, S // 16))
    edges = np.quantile(a0_sample, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    bins = np.clip(np.digitize(a0_sample, edges) - 1, 0, nbins - 1)
    cohort_meds = []
    for b in range(nbins):
        sel = bins == b
        if sel.sum() < 4:
            continue
        d = div[sel]
        finite = d[np.isfinite(d)]
        med = float(np.median(finite)) if finite.size else 0.0
        mad = float(np.median(np.abs(finite - med))) if finite.size else 0.0
        cohort_meds.append(med)
        bar = max(med + K_MAD * mad, DIV_FLOOR)
        flags |= sel & ((div > bar) | ~np.isfinite(div))
    systematic = bool(cohort_meds and np.median(cohort_meds) > FP32_ENVELOPE)
    return {"n": int(S), "div": div, "flags": flags,
            "n_flagged": int(flags.sum()),
            "cohort_median": float(np.median(cohort_meds)) if cohort_meds else 0.0,
            "max_div": float(np.max(div[np.isfinite(div)])) if np.isfinite(div).any() else float("inf"),
            "systematic_fault": systematic}


def fate_audit(amps_resolved, gpu_fates, mats, model, a, nu, n,
               boundary, margin=0.5) -> dict:
    """Fate-level redundancy audit: the mechanism that directly protects A*.
    Re-resolve a sample of resolved lanes' launch amplitudes by the independent
    fp64 path (swarm_m1.resolve_np, gate-validated) and compare FATES.
    Disagreements NEAR the fp64 boundary are the known fp32 shift (physics of
    the instrument, not corruption); disagreements FAR from it are the real
    SDC/systematic signal. amps_resolved, gpu_fates: arrays; boundary: current
    fp64 A* estimate."""
    from swarm_m1 import resolve_np
    f64, _ = resolve_np(np.asarray(amps_resolved, float), mats, nu, model, n,
                        ic="cos2", a=a)
    disagree = (f64 != gpu_fates) & (f64 >= 0)
    far = np.abs(amps_resolved - boundary) > margin
    return {"n": int(len(amps_resolved)),
            "disagreements": int(disagree.sum()),
            "near_boundary": int((disagree & ~far).sum()),   # known fp32
            "far_from_boundary": int((disagree & far).sum())}  # SDC signal


if __name__ == "__main__":
    # calibration + positive control against the class M1 structurally misses
    import sys
    import numpy as np
    import torch
    sys.path.insert(0, ".")
    from swarm_m1 import build_mats, TorchEngine, Ledger

    n = 128
    Hm, Lm, Dm, Gm = build_mats(n)
    mats = {"H": Hm, "L": Lm, "D": Dm, "G": Gm}
    MODEL, A = "gclm", 0.9

    def live_sample(seed):
        eng = TorchEngine(n, mats, 1.0, MODEL, "mps", a=A)
        rng = np.random.default_rng(seed)
        led = Ledger(rng, 8.0, 32.0)
        amps, _ = led.sample(256)
        w = eng.ic(amps, "cos2")
        t = torch.zeros(256, device=eng.dev)
        for _ in range(120):                # march to a live mid-flight state
            w, t, _, _ = eng.macro_step(w, t)
            m = w.abs().amax(1)
            bad = (~torch.isfinite(m)) | (m > 1e3) | ((m < 3.2) & (t > 0.5))
            if bad.any():                   # keep the sample fully live
                idx = torch.where(bad)[0]
                an, _ = led.sample(len(idx))
                w[idx] = eng.ic(an, "cos2"); t[idx] = 0.0; amps[idx.cpu()] = an
        return eng, w, amps

    # (1) CLEAN calibration
    eng, w, amps = live_sample(0)
    r = shadow_audit(eng, w, amps, mats, MODEL, A)
    print(f"CLEAN : n={r['n']} flagged={r['n_flagged']} "
          f"cohort_median_div={r['cohort_median']:.3e} max_div={r['max_div']:.3e} "
          f"systematic={r['systematic_fault']}")

    # (2) POSITIVE CONTROL: consistent-ALU SDC (the exact class the in-loop M1
    #     wire missed). Sweep fault size; CPU fp64 path is clean -> caught in
    #     proportion to fault x regime-amplification, zero false positives.
    for fault in (1.03, 1.01, 1.003):
        eng2, w2, amps2 = live_sample(1)
        victims = list(range(0, 256, 8))   # 32 lanes across the regime spread
        orig = eng2.rhs
        def corrupt(x, _o=orig, _v=victims, _f=fault):
            out = _o(x).clone()
            out[_v] *= _f
            return out
        eng2.rhs = corrupt
        r2 = shadow_audit(eng2, w2, amps2, mats, MODEL, A)
        vflag = [v for v in victims if r2["flags"][v]]
        clean_idx = [i for i in range(256) if i not in victims]
        fp = int(r2["flags"][clean_idx].sum())
        print(f"SDC {(fault-1)*100:.1f}% : caught {len(vflag)}/{len(victims)} "
              f"victims | false_positives {fp}/{len(clean_idx)} | "
              f"victim_div p50/max {np.percentile(r2['div'][victims], 50):.2e}/"
              f"{np.max(r2['div'][victims]):.2e}")

    # (3) SYSTEMATIC fault: ALL lanes corrupted -> cohort-outlier test blind
    #     (no clean peers), but the systematic-envelope check must fire.
    eng3, w3, amps3 = live_sample(2)
    o3 = eng3.rhs
    eng3.rhs = lambda x, _o=o3: _o(x).clone() * 1.02
    r3 = shadow_audit(eng3, w3, amps3, mats, MODEL, A)
    print(f"SYSTEMATIC (all lanes +2%): systematic_fault={r3['systematic_fault']} "
          f"cohort_median_div={r3['cohort_median']:.2e} (envelope {FP32_ENVELOPE})")

    # (4) FATE-LEVEL audit smoke: re-resolve amplitudes in fp64, split
    #     disagreements into known-fp32 (near boundary) vs SDC (far).
    amps_r = np.linspace(10.0, 30.0, 40)
    engf = TorchEngine(n, mats, 1.0, MODEL, "mps", a=A)
    gpu_f, _ = engf.resolve(amps_r, ic="cos2")
    fr = fate_audit(amps_r, gpu_f, mats, MODEL, A, 1.0, n, boundary=18.0)
    print(f"FATE audit: {fr['disagreements']} disagreements "
          f"({fr['near_boundary']} near-boundary=known-fp32, "
          f"{fr['far_from_boundary']} far=SDC-signal) over n={fr['n']}")
