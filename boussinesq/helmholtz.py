#!/usr/bin/env python
"""Fast, well-conditioned Helmholtz solver: (D^2 - alpha) u = f, u(+-1)=0.

The linear-solver upgrade the wall-geometry finding needs. bq2's dense
Chebyshev collocation solve is O(N^2)/mode and ill-conditioned (~N^4, losing
~10 of fp64's 16 digits at N=384) -- the bottleneck at N>=1024 and a possible
contributor to the theta^2-break that caps the conserved window.

Method: Chebyshev-Galerkin with the Shen Dirichlet basis phi_k = T_k - T_{k+2}
(each satisfies phi_k(+-1)=0). In this basis the mass matrix is pentadiagonal
and the stiffness is sparse-upper-triangular, so (Stiff - alpha*Mass) is BANDED
-> O(N) banded solve, and conditioning is ~N^2 not ~N^4. Built from the exact
Chebyshev coefficient-differentiation recurrence (reliable, mechanical); the
GATE against dense collocation is the arbiter of correctness.
"""
from __future__ import annotations

import numpy as np
import scipy.fft
from scipy.linalg import lu_factor, lu_solve, solve_banded


def cheb_coeff_diff(N: int) -> np.ndarray:
    """Matrix mapping Chebyshev-T coefficients of u to those of u' (N x N)."""
    D = np.zeros((N, N))
    for j in range(1, N):
        b = np.zeros(N + 2)
        a = np.zeros(N + 2)
        a[j] = 1.0
        for k in range(N - 1, -1, -1):
            b[k] = b[k + 2] + 2 * (k + 1) * a[k + 1]
        b[0] *= 0.5
        D[:, j] = b[:N]
    return D


def cheb_pts(N: int) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto points on [-1, 1], x_i = cos(pi i/(N-1))."""
    return np.cos(np.pi * np.arange(N) / (N - 1))


def vals_to_coeffs(v: np.ndarray) -> np.ndarray:
    """Grid values on CGL points -> Chebyshev-T coefficients (DCT-I)."""
    N = len(v)
    c = scipy.fft.dct(v, type=1) / (N - 1)
    c[0] *= 0.5
    c[-1] *= 0.5
    return c


def coeffs_to_vals(c: np.ndarray) -> np.ndarray:
    N = len(c)
    c2 = c.copy()
    c2[0] *= 2
    c2[-1] *= 2
    return 0.5 * scipy.fft.dct(c2, type=1)


class ShenHelmholtz:
    """Solve (D^2 - alpha_m) u = f, u(+-1)=0, for a set of alpha_m, in the
    Shen Chebyshev-Galerkin Dirichlet basis. Coefficient space in y."""

    def __init__(self, N: int, alphas: np.ndarray):
        self.N = N
        Dc = cheb_coeff_diff(N)
        D2 = Dc @ Dc                                  # T-coeff 2nd-derivative
        # Shen basis map Phi: Shen-coeffs (N-2) -> T-coeffs (N)
        Phi = np.zeros((N, N - 2))
        for k in range(N - 2):
            Phi[k, k] = 1.0
            Phi[k + 2, k] = -1.0
        h = np.full(N, np.pi / 2)                     # Chebyshev-weight norms
        h[0] = np.pi
        H = np.diag(h)
        self.Slap = Phi.T @ H @ D2 @ Phi              # (phi_j, phi_k'')_w
        self.M = Phi.T @ H @ Phi                      # (phi_j, phi_k)_w
        self.R = Phi.T @ H                            # RHS projector (N-2 x N)
        self.Phi = Phi
        # pre-factorize (Slap - alpha M) per distinct alpha
        self.facs = {float(a): lu_factor(self.Slap - a * self.M) for a in alphas}

    def solve_coeffs(self, fcoef: np.ndarray, alpha: float) -> np.ndarray:
        """f in T-coeffs -> u in T-coeffs (satisfying the BCs)."""
        rhs = self.R @ fcoef
        uhat = lu_solve(self.facs[float(alpha)], rhs)
        return self.Phi @ uhat


# ---------------------------------------------------------------- gate + bench
def dense_helmholtz_solve(f_vals: np.ndarray, alpha: float, x: np.ndarray,
                          D2d: np.ndarray) -> np.ndarray:
    """Reference: dense Chebyshev collocation (bq2's method) for the gate."""
    N = len(x)
    A = D2d - alpha * np.eye(N)
    A[0, :] = 0.0; A[0, 0] = 1.0
    A[-1, :] = 0.0; A[-1, -1] = 1.0
    rhs = f_vals.copy(); rhs[0] = 0.0; rhs[-1] = 0.0
    return lu_solve(lu_factor(A), rhs)


def dense_D2(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense collocation 2nd-derivative matrix (Trefethen) + CGL points."""
    if N == 1:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N) / (N - 1))
    c = np.hstack([2.0, np.ones(N - 2), 2.0]) * (-1.0) ** np.arange(N)
    X = np.tile(x, (N, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N))
    D -= np.diag(D.sum(1))
    return D @ D, x


def gate() -> None:
    print("=== Helmholtz solver gate: Shen-Galerkin vs dense collocation ===")
    for N in (48, 96, 192):
        D2d, x = dense_D2(N)
        rng = np.random.default_rng(0)
        alphas = np.array([0.0, 1.0, 25.0, 400.0])
        shen = ShenHelmholtz(N, alphas)
        worst = 0.0
        for alpha in alphas:
            # manufactured smooth Dirichlet solution
            u_ex = (1 - x ** 2) * np.cos(3 * x) * np.exp(0.5 * x)
            f = D2d @ u_ex - alpha * u_ex          # exact RHS on the grid
            u_dense = dense_helmholtz_solve(f, alpha, x, D2d)
            u_shen = coeffs_to_vals(shen.solve_coeffs(vals_to_coeffs(f), alpha))
            e_shen = np.abs(u_shen - u_ex).max()
            e_dense = np.abs(u_dense - u_ex).max()
            worst = max(worst, e_shen)
            if N == 192 and alpha in (0.0, 400.0):
                print(f"  N={N} alpha={alpha:6.0f}: shen err {e_shen:.2e}  "
                      f"dense err {e_dense:.2e}")
        # conditioning comparison
        cond_dense = np.linalg.cond(D2d - 25 * np.eye(N))
        cond_shen = np.linalg.cond(shen.Slap - 25 * shen.M)
        print(f"  N={N}: worst shen err {worst:.2e} | cond dense "
              f"{cond_dense:.1e} vs shen {cond_shen:.1e} "
              f"({cond_dense/cond_shen:.0f}x better)")
        assert worst < 1e-9, f"GATE FAIL N={N}: shen err {worst:.2e}"
    print("Helmholtz gate PASS (shen == dense to <1e-9, far better conditioned)")


if __name__ == "__main__":
    gate()
