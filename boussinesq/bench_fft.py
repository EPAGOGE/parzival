#!/usr/bin/env python
"""bench_fft.py -- measured 2D transform costs on this machine (era B0 pre-work).

This is BENCH harness, not engine code: wall-clock timing lives here and only
here. Numbers feed HARNESS.md; re-run after any scipy/torch/OS bump and re-date
the doc (meter-era discipline: the meter is part of the measurement).

Run: /Users/epagogellc/parzival/.venv/bin/python /Users/epagogellc/parzival/boussinesq/bench_fft.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy import fft as sfft

SIZES = (128, 256, 512)
BATCH = 16  # candidate lane batch for the fp64 swarm tier


def bench(fn, reps: int, warm: int = 3) -> dict:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return {"median_ms": round(float(np.median(ts)) * 1e3, 4),
            "min_ms": round(float(np.min(ts)) * 1e3, 4),
            "reps": reps}


def main() -> None:
    out: dict = {"scipy_fp64": {}, "torch_mps_fp32": {}}
    rng = np.random.default_rng(0)

    # ---------------- scipy fp64 sine/cosine transforms (the primary engine)
    for n in SIZES:
        x = rng.standard_normal((n, n))
        xb = rng.standard_normal((BATCH, n, n))
        reps = max(20, int(4e7 // (n * n)))
        row = {
            "dstn2_fwd": bench(lambda: sfft.dstn(x, type=2), reps),
            "dstn2_inv": bench(lambda: sfft.idstn(sfft.dstn(x, type=2), type=2),
                               reps // 2),
            "dctn2_fwd": bench(lambda: sfft.dctn(x, type=2), reps),
            "dstn1_fwd": bench(lambda: sfft.dstn(x, type=1), reps),
            f"dstn2_fwd_batch{BATCH}": bench(
                lambda: sfft.dstn(xb, type=2, axes=(-2, -1)), max(10, reps // BATCH)),
            f"dstn2_fwd_batch{BATCH}_workers8": bench(
                lambda: sfft.dstn(xb, type=2, axes=(-2, -1), workers=8),
                max(10, reps // BATCH)),
            "dstn2_fwd_workers8": bench(
                lambda: sfft.dstn(x, type=2, workers=8), reps),
        }
        out["scipy_fp64"][n] = row

    # ---------------- torch MPS fp32 rfft2 (candidate fast tier -- measurement only)
    import torch
    dev = torch.device("mps")
    for n in SIZES:
        xt = torch.randn(n, n, device=dev, dtype=torch.float32)
        xtb = torch.randn(BATCH, n, n, device=dev, dtype=torch.float32)
        reps = max(20, int(4e7 // (n * n)))

        def f_single():
            y = torch.fft.rfft2(xt)
            z = torch.fft.irfft2(y, s=(n, n))
            torch.mps.synchronize()
            return z

        def f_batch():
            y = torch.fft.rfft2(xtb)
            z = torch.fft.irfft2(y, s=(n, n))
            torch.mps.synchronize()
            return z

        try:
            row = {"rfft2_pair": bench(f_single, reps),
                   f"rfft2_pair_batch{BATCH}": bench(f_batch, max(10, reps // BATCH))}
        except Exception as exc:  # MPS FFT support is version-dependent: record, don't guess
            row = {"error": repr(exc)}
        out["torch_mps_fp32"][n] = row

    # ---------------- derived: steps/s and wall time to t=3
    # Transform counts per RK4 step: T=14 (task's low-count formulation,
    # combined/conservative-form transforms) and T=40 (straight pseudo-spectral
    # Boussinesq: ~10 transforms per RHS x 4 stages). Report BOTH.
    # dt(N): CFL-style nominal dt = 1e-3 * (128/N); steps to t=3 = 3/dt.
    derived = {}
    for n in SIZES:
        t_ms = out["scipy_fp64"][n]["dstn2_fwd"]["median_ms"]
        tb_ms = out["scipy_fp64"][n][f"dstn2_fwd_batch{BATCH}_workers8"]["median_ms"] / BATCH
        dt = 1e-3 * (128 / n)
        steps_t3 = int(3 / dt)
        row = {}
        for tag, per in (("single", t_ms), (f"batch{BATCH}_workers8_perlane", tb_ms)):
            for T in (14, 40):
                sps = 1e3 / (T * per)
                row[f"{tag}_T{T}"] = {
                    "per_transform_ms": round(per, 4),
                    "steps_per_s": round(sps, 1),
                    "dt": dt, "steps_to_t3": steps_t3,
                    "wall_to_t3_s": round(steps_t3 / sps, 1)}
        derived[n] = row
    out["derived_rk4"] = derived

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
