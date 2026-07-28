#!/usr/bin/env python
"""Solver LAYER: run N independent Poisson/Helmholtz solvers on the SAME solve
and turn their (dis)agreement into a granular per-mode / per-location trust map.

Not "dense OR Shen" -- a bank. The same redundancy principle as the shadow
auditor, applied to the linear solve. With >=3 independent methods the
disagreement is TRIANGULABLE: 2 solvers tell you they differ, 3 tell you WHICH
is the outlier and WHERE (which Chebyshev locations / which Fourier modes, at
which time). That localization IS an AMR map -- it points at exactly the part
of the corner that needs refinement, which is what the pod run must know.

Bank members (each solves (D^2 - alpha) u = f, u(+-1)=0 on the CGL grid):
  dense  -- Chebyshev collocation (bq2's method; spectral, cond ~1e17)
  shen   -- Chebyshev-Galerkin Shen basis (spectral, cond ~1e3)
  fd     -- non-uniform 2nd-order finite difference (DIFFERENT family; fails
            by truncation, not aliasing/conditioning -- the independent check
            that catches a spectral-specific pathology both Chebyshev methods
            could share)
Plug in more (Legendre-Galerkin, ultraspherical, Dedalus) via add_solver.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve_banded

from helmholtz import (ShenHelmholtz, dense_D2, dense_helmholtz_solve,
                       vals_to_coeffs, coeffs_to_vals)


def make_dense(N: int, alphas):
    D2d, x = dense_D2(N)
    return lambda f, a: dense_helmholtz_solve(f, a, x, D2d), x


def make_shen(N: int, alphas):
    shen = ShenHelmholtz(N, np.asarray(alphas))
    return lambda f, a: coeffs_to_vals(shen.solve_coeffs(vals_to_coeffs(f), a)), None


def make_fd(N: int, alphas):
    """Non-uniform 2nd-order FD on the CGL grid: tridiagonal (D2 - a), u=0 ends.
    Independent discretization family -- truncation error, not conditioning."""
    _, x = dense_D2(N)
    xs = x[::-1]                                    # ascending for clarity
    h = np.diff(xs)
    facs = {}

    def solve(f, a):
        fr = f[::-1]
        lo = np.zeros(N); di = np.zeros(N); up = np.zeros(N)
        di[0] = di[-1] = 1.0                        # Dirichlet rows
        rhs = fr.copy(); rhs[0] = 0.0; rhs[-1] = 0.0
        for i in range(1, N - 1):
            hm, hp = h[i - 1], h[i]
            lo[i] = 2.0 / (hm * (hm + hp))
            di[i] = -2.0 / (hm * hp) - a
            up[i] = 2.0 / (hp * (hm + hp))
        ab = np.zeros((3, N))
        ab[0, 1:] = up[:-1]; ab[1] = di; ab[2, :-1] = lo[1:]
        u = solve_banded((1, 1), ab, rhs)
        return u[::-1]
    return solve, None


class SolverBank:
    def __init__(self, N: int, alphas):
        self.N = N
        self.solvers = {}
        for name, mk in (("dense", make_dense), ("shen", make_shen),
                         ("fd", make_fd)):
            fn, _ = mk(N, alphas)
            self.solvers[name] = fn

    def add_solver(self, name, fn):
        self.solvers[name] = fn

    def audit(self, f: np.ndarray, alpha: float) -> dict:
        """Run every solver on (f, alpha). Return the consensus (median),
        per-solver deviation from it, and the per-location disagreement
        (the granular map). fd is 2nd-order so a smooth-field baseline
        deviation ~1/N^2 is expected -- EXCESS over that is the real signal."""
        us = {k: v(f, alpha) for k, v in self.solvers.items()}
        stack = np.array(list(us.values()))
        consensus = np.median(stack, axis=0)
        scale = max(np.abs(consensus).max(), 1e-30)
        dev = {k: float(np.abs(us[k] - consensus).max() / scale) for k in us}
        outlier = max(dev, key=dev.get)
        per_loc = np.abs(stack - consensus).max(axis=0) / scale  # granular map
        return {"consensus": consensus, "dev": dev, "outlier": outlier,
                "outlier_dev": dev[outlier], "per_loc": per_loc,
                "max_loc": int(np.argmax(per_loc))}


# ---------------------------------------------------------------- demo / gate
def demo() -> None:
    print("=== solver bank: granular cross-agreement ===")
    N = 128
    _, x = dense_D2(N)
    alphas = [0.0, 100.0]
    bank = SolverBank(N, alphas)

    # (1) smooth field: all spectral agree to roundoff; fd tracks to O(1/N^2)
    u_ex = (1 - x ** 2) * np.cos(4 * x)
    D2d, _ = dense_D2(N)
    f = D2d @ u_ex - 100.0 * u_ex
    r = bank.audit(f, 100.0)
    print(f"smooth : dev {{k: round(v,2e) ...}} = "
          + ", ".join(f"{k} {v:.1e}" for k, v in r['dev'].items())
          + f" | outlier {r['outlier']}")

    # (2) STRESSED: high-k content packed near the wall (x~-1), where dense
    #     Chebyshev conditioning is worst. The bank should isolate dense as the
    #     outlier and localize the disagreement to the near-wall region.
    u_st = (1 - x ** 2) * np.cos(40 * x) * np.exp(-8 * (x + 1) ** 2)
    f2 = D2d @ u_st - 100.0 * u_st
    r2 = bank.audit(f2, 100.0)
    xstar = x[r2["max_loc"]]
    print(f"stressed: " + ", ".join(f"{k} {v:.1e}" for k, v in r2['dev'].items())
          + f" | OUTLIER={r2['outlier']} localized at x={xstar:+.3f} "
          f"({'near wall' if abs(xstar+1) < 0.3 or abs(xstar-1) < 0.3 else 'interior'})")
    # the shen/fd consensus should be the accurate one
    e_cons = np.abs(r2["consensus"] - u_st).max()
    e_dense = np.abs(bank.solvers["dense"](f2, 100.0) - u_st).max()
    print(f"         consensus err {e_cons:.2e} vs dense-alone err {e_dense:.2e} "
          f"({e_dense/max(e_cons,1e-30):.0f}x -- the bank is more accurate than "
          f"its worst member)")


if __name__ == "__main__":
    demo()
