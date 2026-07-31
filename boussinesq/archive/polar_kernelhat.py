"""Kernel-hat architecture, stage-1 build: the compactified far-field panel.

Derivation and gates: KERNELHAT_DERIVATION.md (verified in sympy 2026-07-28).
This file is built bottom-up with a testable unit at each layer; nothing in
polar_cornerreg.py is touched.

LAYER 1 (this commit): the compactified panel [xi2, inf) via xi = xi2 + L*t/(1-t),
Chebyshev-Lobatto in t INCLUDING t = 1 (xi = inf), with first and second
xi-derivative matrices assembled by the chain rule

    d/dxi   = ((1-t)^2 / L) d/dt
    d2/dxi2 = ((1-t)^4 / L^2) d2/dt2  -  (2 (1-t)^3 / L^2) d/dt

At t = 1 both factors vanish: derivative rows there return 0 identically, which is
CORRECT for plateau functions (the hatted fields' xi-derivatives vanish at
infinity faster than any power) and is exactly the degeneracy the divided
transport rows and the surviving elliptic row are designed around.

Self-test: python polar_kernelhat.py  -- differentiates the two structures the
far field is made of (e^(a0*xi) and 1/(1+xi) plateaus) and reports max errors on
the finite nodes. Bar: spectral accuracy (<1e-8) on the correction structure.
"""
from __future__ import annotations

import numpy as np


def cheb_lobatto(n: int):
    """Chebyshev-Gauss-Lobatto nodes on [0,1] ascending, with D (dense)."""
    k = np.arange(n + 1)
    xc = np.cos(np.pi * k / n)             # [1 .. -1]
    t = (1.0 - xc) / 2.0                   # ascending [0 .. 1]
    c = np.ones(n + 1); c[0] = c[-1] = 2.0
    c *= (-1.0) ** k
    X = np.tile(xc, (n + 1, 1)).T
    dX = X - X.T + np.eye(n + 1)
    D = np.outer(c, 1.0 / c) / dX
    D -= np.diag(D.sum(axis=1))
    # chain to ascending t in [0,1]: d/dt = -2 d/dxc
    Dt = -2.0 * D
    return t, Dt


class CompactPanel:
    """[xi2, inf) with algebraic map xi = xi2 + L*t/(1-t), t in [0,1]."""

    def __init__(self, xi2: float, L: float, deg: int):
        self.xi2, self.L, self.deg = float(xi2), float(L), int(deg)
        self.t, Dt = cheb_lobatto(deg)
        Dt2 = Dt @ Dt
        omt = 1.0 - self.t
        with np.errstate(divide="ignore"):
            self.xi = np.where(self.t < 1.0, xi2 + L * self.t / omt, np.inf)
        f1 = omt ** 2 / L                       # dxi -> dt factor
        self.Dxi = f1[:, None] * Dt
        self.Dxi2 = (omt ** 4 / L ** 2)[:, None] * Dt2 \
            - (2.0 * omt ** 3 / L ** 2)[:, None] * Dt
        # endpoint rows are exactly zero by construction (omt = 0)


def _selftest():
    a0 = -0.34471228737239
    for deg, L in [(16, 10.0), (24, 10.0), (24, 20.0)]:
        P = CompactPanel(15.0, L, deg)
        fin = P.t < 1.0
        xi = P.xi[fin]

        # structure 1: the far-field correction e^(a0 xi)
        f = np.zeros_like(P.t); f[fin] = np.exp(a0 * xi)      # f(inf)=0
        df = P.Dxi @ f; d2f = P.Dxi2 @ f
        e1 = np.abs(df[fin] - a0 * np.exp(a0 * xi)).max()
        e2 = np.abs(d2f[fin] - a0 ** 2 * np.exp(a0 * xi)).max()

        # structure 2: a plateau approached like 1/(1+xi)  (log-kernel remnant)
        g = np.zeros_like(P.t)
        g[fin] = 1.0 / (1.0 + xi); gval_inf = 0.0
        g[~fin] = gval_inf
        dg = P.Dxi @ g
        e3 = np.abs(dg[fin] + 1.0 / (1.0 + xi) ** 2).max()

        # endpoint rows identically zero
        e4 = max(abs(df[-1]), abs(d2f[-1]), abs(dg[-1]))
        print(f"deg={deg:3d} L={L:5.1f}  |d(e^a0x)err| {e1:.2e}  "
              f"|d2 err| {e2:.2e}  |d(1/(1+x))err| {e3:.2e}  endpoint {e4:.1e}")


if __name__ == "__main__":
    _selftest()
