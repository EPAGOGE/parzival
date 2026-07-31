#!/usr/bin/env python
"""BREAKER attack 2: what is the 2/3 rule actually suppressing?

bq.py exposes no dealias toggle, so this constructs the aliased products
manually at N=64 on a hot state (A=6 evolved adaptively until tails are
loaded) and quantifies:

  (a) EXACTNESS: the engine's masked pseudo-spectral product vs the exact
      product computed on a 2N fine grid (zero-padded coefficients, no
      aliasing possible: input freqs <= K=42, product freqs <= 84 < 128).
      In-band (<=K) disagreement should be roundoff if the 2/3 rule works.
  (b) JUNK: the coefficient content of the UNMASKED N-grid product above K
      (true high-freq content + aliased reflections) that the mask kills.
  (c) IN-BAND ALIASING WITHOUT THE RULE: an unmasked engine variant
      (MASK=1, same K for dt/tails) -- single-RHS in-band error vs exact,
      and a 400-step adaptive integration comparing theta^2 drift
      masked vs unmasked.

New file; bq.py untouched. Run:
  /Users/epagogellc/parzival/.venv/bin/python \
      /Users/epagogellc/parzival/boussinesq/breaker_dealias.py
"""
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bq import (BQ, theta0_hat, to_ss, from_ss, to_cs, from_cs,  # noqa: E402
                from_sc, from_cc, d_sin2cos, d_cos2sin)

N = 64
M = 128          # fine grid: product of two freq<=42 series has freq<=84 < 128
A = 6.0


def embed(F: np.ndarray, M: int) -> np.ndarray:
    """Zero-pad ortho parity coefficients N->M. Valid because exceptional
    slots (sine freq N, cos slot 0 aside) are either zero under the 2/3 mask
    or scale as sqrt(N) like the shared band; ortho coefs scale sqrt(N)/axis."""
    n = F.shape[0]
    G = np.zeros((M, M))
    G[:n, :n] = F
    return G * (M / n)


def selftest(eng: BQ) -> float:
    """Embedding fidelity: analytic product of two low modes vs fine grid."""
    x1, x2 = eng.x[:, None], eng.x[None, :]
    f = np.sin(3 * x1) * np.sin(5 * x2)
    F = to_ss(f)
    engM = BQ(M)
    xf1, xf2 = engM.x[:, None], engM.x[None, :]
    fine = from_ss(embed(F, M))
    return float(np.abs(fine - np.sin(3 * xf1) * np.sin(5 * xf2)).max())


class BQnomask(BQ):
    """Same engine, 2/3 product mask disabled (masks = 1). K untouched so the
    dt law and tail bands are identical -- isolates the dealiasing rule."""

    def __init__(self, N: int, **kw):
        super().__init__(N, **kw)
        self.MASK_SS = np.ones_like(self.MASK_SS)
        self.MASK_CS = np.ones_like(self.MASK_CS)


def hot_state(engine_cls, steps: int) -> tuple:
    eng = engine_cls(N)
    w = np.zeros((N, N))
    th = theta0_hat(eng, A) * eng.MASK_CS if engine_cls is BQ else \
        theta0_hat(BQ(N), A)  # BQ path already masks inside theta0_hat
    if engine_cls is not BQ:
        th = theta0_hat(BQnomask(N), A)   # unmasked IC for the unmasked engine
    for _ in range(steps):
        w, th, _, _ = eng.step(w, th)
    return eng, w, th


def exact_products(eng: BQ, w_hat, th_hat):
    """Dealiased-by-construction advection products via the M=2N fine grid,
    truncated back to the N-grid band. Mirrors BQ.rhs term for term."""
    psi_hat = -w_hat / eng.LAM
    u1_hat, u2_hat = -d_sin2cos(psi_hat, 1), d_sin2cos(psi_hat, 0)
    wx1_hat, wx2_hat = d_sin2cos(w_hat, 0), d_sin2cos(w_hat, 1)
    tx1_hat, tx2_hat = d_cos2sin(th_hat, 0), d_sin2cos(th_hat, 1)
    u1f, u2f = from_sc(embed(u1_hat, M)), from_cs(embed(u2_hat, M))
    wx1f, wx2f = from_cs(embed(wx1_hat, M)), from_sc(embed(wx2_hat, M))
    tx1f, tx2f = from_ss(embed(tx1_hat, M)), from_cc(embed(tx2_hat, M))
    advw_M = to_ss(u1f * wx1f + u2f * wx2f) * (N / M)
    advt_M = to_cs(u1f * tx1f + u2f * tx2f) * (N / M)
    return advw_M[:N, :N], advt_M[:N, :N]


def grid_products(eng: BQ, w_hat, th_hat):
    """The engine's own collocation products on the N grid, UNMASKED."""
    psi_hat = -w_hat / eng.LAM
    u1_hat, u2_hat = -d_sin2cos(psi_hat, 1), d_sin2cos(psi_hat, 0)
    u1, u2 = from_sc(u1_hat), from_cs(u2_hat)
    wx1, wx2 = from_cs(d_sin2cos(w_hat, 0)), from_sc(d_sin2cos(w_hat, 1))
    tx1, tx2 = from_ss(d_cos2sin(th_hat, 0)), from_cc(d_sin2cos(th_hat, 1))
    return to_ss(u1 * wx1 + u2 * wx2), to_cs(u1 * tx1 + u2 * tx2)


def main() -> None:
    eng = BQ(N)
    print(f"embedding self-test max err: {selftest(eng):.3e}  (want ~1e-15)")

    # hot masked state at N=64: step until tails loaded (~t past trust)
    _, w, th = hot_state(BQ, 260)
    b = eng.budgets(w, th)
    print(f"hot state: tail_w {b['tail_w']:.3e} tail_th {b['tail_th']:.3e} "
          f"K={eng.K}")

    advw_grid, advt_grid = grid_products(eng, w, th)
    advw_ex, advt_ex = exact_products(eng, w, th)
    mss, mcs = eng.MASK_SS.astype(bool), eng.MASK_CS.astype(bool)

    # (a) in-band exactness of the engine's masked product
    for nm, g, ex, m in (("adv_w", advw_grid, advw_ex, mss),
                         ("adv_t", advt_grid, advt_ex, mcs)):
        inband = float(np.abs((g - ex))[m].max())
        scale = float(np.abs(ex[m]).max())
        # (b) what the mask kills: unmasked N-grid product content above K
        junk = float(np.sqrt(np.sum(g[~m] ** 2) / np.sum(g[m] ** 2)))
        # how much of that junk is ALIASED (not true content): compare with
        # exact product's own above-K content on the N-band
        true_hi = float(np.sqrt(np.sum(ex[~m] ** 2) / np.sum(ex[m] ** 2)))
        print(f"{nm}: in-band |masked - exact| max {inband:.3e} "
              f"(rel {inband / scale:.3e}); above-K junk/inband L2 "
              f"{junk:.3e} (true content {true_hi:.3e}, rest is alias+trunc)")

    # (c) run the unmasked engine head-to-head, adaptive dt
    for cls, tag in ((BQ, "masked  "), (BQnomask, "unmasked")):
        e = cls(N)
        wte = np.zeros((N, N))
        the = theta0_hat(e, A)
        q0 = float(np.sum(the ** 2))
        t = 0.0
        status = "ok"
        for s in range(400):
            w2, t2, dt, _ = e.step(wte, the)
            if not (np.isfinite(w2).all() and np.isfinite(t2).all()):
                status = f"NONFINITE at step {s}"
                break
            wte, the = w2, t2
            t += dt
        drift = abs(float(np.sum(the ** 2)) - q0) / q0
        bb = BQ(N).budgets(wte, the)
        print(f"{tag}: 400 adaptive steps -> t={t:.4f} {status}; "
              f"theta^2 drift {drift:.3e}; tail_w {bb['tail_w']:.3e}")


if __name__ == "__main__":
    main()
