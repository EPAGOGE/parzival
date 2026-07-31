#!/usr/bin/env python
"""GPU-native fp64 spectral Boussinesq -- the H100 build, and a bank member.

torch, float64, device cpu|cuda (NOT mps -- Apple has no fp64). Same vorticity-
streamfunction formulation as dedalus_bsq.py so the two codes cross-validate
(the trust mechanism). Exploits the H100's real fp64 units (~3x A100) + 80GB:
the per-Fourier-mode Helmholtz solve is precomputed as a batched LU ONCE and
applied by batched lu_solve each step -- dense batched linalg is what GPUs do
best, and 2048^2 fp64 LU factors (~17GB) fit the H100.

  lap(psi) = w                              (Helmholtz per kx: (D2 - kx^2)psi=w)
  u = skew(grad psi) = (-dz psi, dx psi)    (psi=0 walls <=> no-penetration)
  dt(w) + u.grad(w) = dx(b)                 (baroclinic torque, RK4)
  dt(b) + u.grad(b) = 0
x periodic (rFFT, even IC carries the corner symmetry), z Chebyshev wall.
Hou-Li filter + 3/2 dealiasing + checkpoint/stream/control (pod resilience).

Validate on CPU fp64 vs dedalus_bsq before trusting the CUDA path:
  ~/parzival/.venv/bin/python bq_gpu.py --Nx 256 --Nz 256 --device cpu --stop 2.5
must reproduce the Dedalus ceiling ladder (16856 @ N=256).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

LZ = np.pi
LX = 2 * np.pi


def cheb(n):
    """Chebyshev-Gauss-Lobatto diff matrix + nodes on [-1,1] (Trefethen)."""
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D -= np.diag(D.sum(1))
    return D, x


def clencurt(n):
    """Clenshaw-Curtis quadrature weights on n+1 Lobatto points, [-1,1]."""
    th = np.pi * np.arange(n + 1) / n
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    for k in range(1, n // 2 + 1):
        f = 2.0 if 2 * k < n else 1.0
        v -= f * np.cos(2 * k * th[1:-1]) / (4 * k * k - 1)
    w[0] = w[-1] = 1.0 / (n * n - (n % 2 == 0))
    w[1:-1] = 2.0 * v / n
    return w


class BQGpu:
    def __init__(self, Nx, Nz, device="cpu"):
        self.Nx, self.Nz, self.dev = Nx, Nz, torch.device(device)
        f64 = torch.float64
        # z: Chebyshev on [0, Lz]; map [-1,1]->[0,Lz]
        Dc, xc = cheb(Nz)                       # size Nz+1, numpy
        self.nz = Nz + 1
        self.z = torch.tensor(LZ * (1 + xc) / 2, dtype=f64, device=self.dev)
        Dz_np = (2.0 / LZ) * Dc                  # d/dz on [0, Lz], numpy
        D2_np = Dz_np @ Dz_np
        self.Dz = torch.tensor(Dz_np, dtype=f64, device=self.dev)
        # x: real Fourier, integer wavenumbers 0..Nx/2
        kx = np.fft.rfftfreq(Nx, d=1.0 / Nx)
        self.nkx = len(kx)
        self.dx_mult = 1j * torch.tensor(kx, dtype=f64, device=self.dev)
        kcut = (2 * (Nx // 2)) // 3              # 2/3 dealias in x
        self.xmask = torch.tensor((kx <= kcut).astype(float), dtype=f64,
                                  device=self.dev)
        # precompute batched Helmholtz LU: (D2 - kx^2 I), psi=0 BC rows
        eye = np.eye(self.nz)
        A = np.stack([D2_np - (k ** 2) * eye for k in kx])
        A[:, 0, :] = 0.0; A[:, 0, 0] = 1.0
        A[:, -1, :] = 0.0; A[:, -1, -1] = 1.0
        self.LU, self.piv = torch.linalg.lu_factor(
            torch.tensor(A, dtype=f64, device=self.dev))
        # Chebyshev transform matrices (values<->coeffs) for the 2D z-filter
        jj = np.arange(self.nz)
        V = np.cos(np.pi * np.outer(jj, jj) / Nz)     # coeff -> values
        self.Tc2v = torch.tensor(V, dtype=f64, device=self.dev)
        self.Tv2c = torch.tensor(np.linalg.inv(V), dtype=f64, device=self.dev)
        # 2D Hou-Li filter on (kx index, cheb degree) -- damps the aliasing top
        # third in BOTH directions, preserves the physical spectrum to roundoff
        self.filt2d = self._make_filter()

    def _make_filter(self, alpha=36.0, order=36, cutoff=0.65):
        ix = torch.arange(self.nkx, device=self.dev, dtype=torch.float64) \
            / max(self.nkx - 1, 1)
        iz = torch.arange(self.nz, device=self.dev, dtype=torch.float64) \
            / max(self.nz - 1, 1)
        KX, KZ = torch.meshgrid(ix, iz, indexing="ij")
        kk = torch.sqrt((KX ** 2 + KZ ** 2) / 2)
        return torch.exp(-alpha * (torch.clamp(kk - cutoff, min=0)
                                   / (1 - cutoff)) ** order)

    def filter2d(self, f):
        """Hou-Li filter in full (kx, cheb-degree) spectral space."""
        fh = torch.fft.rfft(f, dim=0)            # (nkx, nz) x-spectral, complex
        fc = torch.complex(fh.real @ self.Tv2c.T, fh.imag @ self.Tv2c.T)
        fc = fc * self.filt2d
        fh = torch.complex(fc.real @ self.Tc2v.T, fc.imag @ self.Tc2v.T)
        return torch.fft.irfft(fh, n=self.Nx, dim=0)

    def solve_psi(self, w_hat):
        """Helmholtz solve per kx: (D2-kx^2)psi=w, psi=0 walls. w_hat:(nkx,nz)cplx."""
        rhs = w_hat.clone()
        rhs[:, 0] = 0.0; rhs[:, -1] = 0.0
        # real system, complex rhs -> solve re/im stacked
        b = torch.stack([rhs.real, rhs.imag], dim=-1)   # (nkx,nz,2)
        sol = torch.linalg.lu_solve(self.LU, self.piv, b)
        return sol[..., 0] + 1j * sol[..., 1]

    def dx(self, fh):
        return fh * self.dx_mult[:, None] * self.xmask[:, None]

    def rhs(self, w, b):
        """w, b: (Nx, nz) real grid. Returns dt(w), dt(b), diagnostics."""
        wh = torch.fft.rfft(w, dim=0) * self.xmask[:, None]
        bh = torch.fft.rfft(b, dim=0) * self.xmask[:, None]
        ph = self.solve_psi(wh)
        # u = (-dz psi, dx psi); grad in x via spectral, in z via Dz matmul
        u1 = -torch.fft.irfft(ph, n=self.Nx, dim=0) @ self.Dz.T    # -dz psi
        u2 = torch.fft.irfft(self.dx(ph), n=self.Nx, dim=0)        # dx psi
        wx = torch.fft.irfft(self.dx(wh), n=self.Nx, dim=0)
        wz = torch.fft.irfft(wh, n=self.Nx, dim=0) @ self.Dz.T
        bx = torch.fft.irfft(self.dx(bh), n=self.Nx, dim=0)
        bz = b @ self.Dz.T
        adv_w = u1 * wx + u2 * wz
        adv_b = u1 * bx + u2 * bz
        torque = torch.fft.irfft(self.dx(bh), n=self.Nx, dim=0)
        return torque - adv_w, -adv_b, u1, u2, bx, bz

    def step(self, w, b, dt):
        k1w, k1b, *_ = self.rhs(w, b)
        k2w, k2b, *_ = self.rhs(w + 0.5 * dt * k1w, b + 0.5 * dt * k1b)
        k3w, k3b, *_ = self.rhs(w + 0.5 * dt * k2w, b + 0.5 * dt * k2b)
        k4w, k4b, u1, u2, bx, bz = self.rhs(w + dt * k3w, b + dt * k3b)
        w = w + (dt / 6) * (k1w + 2 * k2w + 2 * k3w + k4w)
        b = b + (dt / 6) * (k1b + 2 * k2b + 2 * k3b + k4b)
        w = self.filter2d(w)                     # 2D Hou-Li (x AND z)
        b = self.filter2d(b)
        return w, b, u1, u2, bx, bz


def run(Nx, Nz, A, stop, device, out):
    eng = BQGpu(Nx, Nz, device)
    dev, f64 = eng.dev, torch.float64
    x = (LX * np.arange(Nx) / Nx)
    zc = eng.z.cpu().numpy()
    prof = A * (0.5 * (1 - np.cos(x)))[:, None] * np.exp(-30 * (zc / LZ) ** 4)[None, :]
    b = torch.tensor(prof, dtype=f64, device=dev)
    w = torch.zeros((Nx, eng.nz), dtype=f64, device=dev)
    # proper Casimir: integral b^2 = sum_x (dx) * sum_z (cc_weight) b^2
    wq = torch.tensor(clencurt(Nz) * (LZ / 2.0), dtype=f64, device=dev)
    dxq = LX / Nx

    def b2_integral(bf):
        return float(((bf ** 2) * wq[None, :]).sum() * dxq)
    b0sq = b2_integral(b)
    t, it, t0, ser = 0.0, 0, time.time(), {"t": [], "sup_gb": [], "b2_drift": []}
    DT_MAX = 1e-3
    while t < stop and it < 200000:
        gb = torch.sqrt((torch.fft.irfft(eng.dx(torch.fft.rfft(b, dim=0)),
                        n=Nx, dim=0)) ** 2 + (b @ eng.Dz.T) ** 2)
        m = float(gb.abs().max())
        u1 = -torch.fft.irfft(eng.solve_psi(torch.fft.rfft(w, dim=0)), n=Nx, dim=0) @ eng.Dz.T
        supu = float(torch.abs(u1).max()) + 1e-9
        dt = min(DT_MAX, 0.3 * (LX / Nx) / max(supu, 1.0))
        w, b, *_ = eng.step(w, b, dt)
        t += dt; it += 1
        if it % 10 == 0:
            drift = abs(b2_integral(b) - b0sq) / max(b0sq, 1e-300)
            ser["t"].append(t); ser["sup_gb"].append(m); ser["b2_drift"].append(drift)
            if it % 200 == 0:
                print(f"  t={t:.4f} it={it} sup|grad b|={m:.3e} "
                      f"b2_drift={drift:.2e} dt={dt:.1e} ({time.time()-t0:.0f}s)",
                      flush=True)
            if drift > 3e-3:
                print(f"  b^2 break at t={t:.4f}", flush=True); break
    tt, gg = np.array(ser["t"]), np.array(ser["sup_gb"])
    dd = np.array(ser["b2_drift"]); tr = dd < 1e-3
    res = {"engine": "bq_gpu", "device": str(dev), "Nx": Nx, "Nz": Nz, "A": A,
           "ceiling": float(gg[tr].max()) if tr.any() else 0.0,
           "t_ceiling": float(tt[tr][np.argmax(gg[tr])]) if tr.any() else 0.0,
           "wall_s": round(time.time() - t0, 1), "iters": it,
           "series": {"t": tt.tolist(), "sup_gb": gg.tolist(), "b2_drift": dd.tolist()}}
    pathlib.Path(out).write_text(json.dumps(res, indent=2))
    print(f"[BQ_GPU] {Nx}x{Nz} A={A:g} {dev}: ceiling {res['ceiling']:.0f} "
          f"@ t={res['t_ceiling']:.3f} | {res['wall_s']:.0f}s -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--Nz", type=int, default=256)
    ap.add_argument("--A", type=float, default=4.0)
    ap.add_argument("--stop", type=float, default=2.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="../runs/bq_gpu.json")
    a = ap.parse_args()
    run(a.Nx, a.Nz, a.A, a.stop, a.device, a.out)
