#!/usr/bin/env python
"""GPU-native fp64 spectral Boussinesq -- CORRECT (dealiased) version.

Fixes bq_gpu.py's early Casimir break. Root cause (confirmed: drift jumps at
t~1.33 as the front sharpens): the nonlinear advection products u.grad(w),
u.grad(b) were formed on the grid and NEVER dealiased, so aliasing at the
sharpening front injected spurious b^2. Dedalus dealiases the products inside
every RHS; this does the same -- 2/3 truncation in BOTH x (Fourier) and z
(Chebyshev) applied to the advection terms -- plus the Hou-Li filter each step.

torch, float64, device cpu|cuda (NOT mps: no fp64). Vorticity-streamfunction,
so matches dedalus_bsq.py. Helmholtz via batched dense LU here (conditioning is
NOT the issue -- proven this session: Shen==dense window). For POD MEMORY
HEADROOM at high N, swap solve_psi to the banded Shen-Galerkin solve
(helmholtz.py) -- dense LU factors are ~17GB at N=2048; the swap is memory-only,
correctness is unchanged (this is the documented pod-headroom step).

Validate on CPU fp64 vs Dedalus: N=128 must conserve b^2 to t~2.4 (grad-b ->
several hundred), N=256 ceiling ~16856. NOT break at grad~22.
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
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D -= np.diag(D.sum(1))
    return D, x


def clencurt(n):
    th = np.pi * np.arange(n + 1) / n
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    for k in range(1, n // 2 + 1):
        f = 2.0 if 2 * k < n else 1.0
        v -= f * np.cos(2 * k * th[1:-1]) / (4 * k * k - 1)
    w[0] = w[-1] = 1.0 / (n * n - (n % 2 == 0))
    w[1:-1] = 2.0 * v / n
    return w


class BQ2Gpu:
    def __init__(self, Nx, Nz, device="cpu"):
        self.Nx, self.Nz, self.dev = Nx, Nz, torch.device(device)
        f64 = torch.float64
        Dc, xc = cheb(Nz)
        self.nz = Nz + 1
        self.z = torch.tensor(LZ * (1 + xc) / 2, dtype=f64, device=self.dev)
        Dz_np = (2.0 / LZ) * Dc
        self.Dz = torch.tensor(Dz_np, dtype=f64, device=self.dev)
        kx = np.fft.rfftfreq(Nx, d=1.0 / Nx)
        self.nkx = len(kx)
        self.dx_mult = 1j * torch.tensor(kx, dtype=f64, device=self.dev)
        # 2/3 dealias masks in BOTH directions
        kxcut = (2 * (Nx // 2)) // 3
        self.xmask = torch.tensor((kx <= kxcut).astype(float), dtype=f64,
                                  device=self.dev)[:, None]
        kzcut = (2 * Nz) // 3
        zm = (np.arange(self.nz) <= kzcut).astype(float)
        self.zmask = torch.tensor(zm, dtype=f64, device=self.dev)[None, :]
        # Chebyshev transforms (values <-> coeffs)
        jj = np.arange(self.nz)
        V = np.cos(np.pi * np.outer(jj, jj) / Nz)
        self.Tc2v = torch.tensor(V, dtype=f64, device=self.dev)
        self.Tv2c = torch.tensor(np.linalg.inv(V), dtype=f64, device=self.dev)
        # 3/2 dealiasing: fine grids Mx (x), Mz+1 (z). Products formed on the
        # fine grid keep FULL coarse resolution AND are alias-free (unlike 2/3
        # truncation, which smears by discarding 1/3 of the modes).
        self.Mx = 3 * Nx // 2
        Mz = 3 * Nz // 2
        self.mzf = Mz + 1
        self.nkxf = self.Mx // 2 + 1
        # z: coarse coeff (nz) -> fine values (Mz+1):  E_up[j,k]=cos(k*pi*j/Mz)
        E_up = np.cos(np.pi * np.outer(np.arange(self.mzf), jj) / Mz)  # (mzf, nz)
        self.E_up = torch.tensor(E_up, dtype=f64, device=self.dev)
        # z: fine values (Mz+1) -> fine coeff (Mz+1)
        jf = np.arange(self.mzf)
        Vf = np.cos(np.pi * np.outer(jf, jf) / Mz)
        self.Tv2cf = torch.tensor(np.linalg.inv(Vf), dtype=f64, device=self.dev)
        # Helmholtz batched LU: (D2 - kx^2), psi=0 walls
        eye = np.eye(self.nz)
        D2 = Dz_np @ Dz_np
        A = np.stack([D2 - (k ** 2) * eye for k in kx])
        A[:, 0, :] = 0.0; A[:, 0, 0] = 1.0
        A[:, -1, :] = 0.0; A[:, -1, -1] = 1.0
        self.LU, self.piv = torch.linalg.lu_factor(
            torch.tensor(A, dtype=f64, device=self.dev))
        self.filt2d = self._filter()

    def _filter(self, alpha=36.0, order=36, cutoff=0.65):
        ix = torch.arange(self.nkx, device=self.dev, dtype=torch.float64) / max(self.nkx - 1, 1)
        iz = torch.arange(self.nz, device=self.dev, dtype=torch.float64) / max(self.nz - 1, 1)
        KX, KZ = torch.meshgrid(ix, iz, indexing="ij")
        kk = torch.sqrt((KX ** 2 + KZ ** 2) / 2)
        return torch.exp(-alpha * (torch.clamp(kk - cutoff, min=0) / (1 - cutoff)) ** order)

    def _zc(self, fh):                       # x-spectral -> full spectral (z coeff)
        return torch.complex(fh.real @ self.Tv2c.T, fh.imag @ self.Tv2c.T)

    def _zv(self, fc):                       # full spectral -> x-spectral (z values)
        return torch.complex(fc.real @ self.Tc2v.T, fc.imag @ self.Tc2v.T)

    def _up(self, f):
        """coarse physical (Nx,nz) -> fine physical (Mx,mzf), 3/2 padded."""
        fh = torch.fft.rfft(f, dim=0)                    # (nkx, nz)
        pad = torch.zeros((self.nkxf, self.nz), dtype=fh.dtype, device=self.dev)
        pad[:fh.shape[0]] = fh
        fx = torch.fft.irfft(pad, n=self.Mx, dim=0) * (self.Mx / self.Nx)
        return (fx @ self.Tv2c.T) @ self.E_up.T          # (Mx, mzf) fine values

    def _down(self, ff):
        """fine physical (Mx,mzf) -> coarse physical (Nx,nz), truncated."""
        fv = (ff @ self.Tv2cf.T)[:, :self.nz] @ self.Tc2v.T   # z fine->coarse
        fh = torch.fft.rfft(fv, dim=0)[:self.nkx] * (self.Nx / self.Mx)
        return torch.fft.irfft(fh, n=self.Nx, dim=0)

    def _prod(self, *pairs):
        """Sum of 3/2-dealiased products: _prod((a,b),(c,d)) = ~(a*b + c*d)."""
        acc = None
        for a, b in pairs:
            p = self._up(a) * self._up(b)
            acc = p if acc is None else acc + p
        return self._down(acc)

    def filt(self, f):
        fh = torch.fft.rfft(f, dim=0)
        fc = self._zc(fh) * self.filt2d
        return torch.fft.irfft(self._zv(fc), n=self.Nx, dim=0)

    def solve_psi(self, w_hat):
        rhs = w_hat.clone()
        rhs[:, 0] = 0.0; rhs[:, -1] = 0.0
        b = torch.stack([rhs.real, rhs.imag], dim=-1)
        sol = torch.linalg.lu_solve(self.LU, self.piv, b)
        return sol[..., 0] + 1j * sol[..., 1]

    def dxf(self, fh):
        return fh * self.dx_mult[:, None] * self.xmask

    def rhs(self, w, b):
        wh = torch.fft.rfft(w, dim=0) * self.xmask
        bh = torch.fft.rfft(b, dim=0) * self.xmask
        ph = self.solve_psi(wh)
        u1 = -torch.fft.irfft(ph, n=self.Nx, dim=0) @ self.Dz.T
        u2 = torch.fft.irfft(self.dxf(ph), n=self.Nx, dim=0)
        wx = torch.fft.irfft(self.dxf(wh), n=self.Nx, dim=0)
        wz = torch.fft.irfft(wh, n=self.Nx, dim=0) @ self.Dz.T
        bx = torch.fft.irfft(self.dxf(bh), n=self.Nx, dim=0)
        bz = torch.fft.irfft(bh, n=self.Nx, dim=0) @ self.Dz.T
        torque = bx
        # 3/2-dealiased advection products: full resolution, alias-free
        adv_w = self._prod((u1, wx), (u2, wz))
        adv_b = self._prod((u1, bx), (u2, bz))
        return torque - adv_w, -adv_b, u1

    def step(self, w, b, dt):
        k1w, k1b, _ = self.rhs(w, b)
        k2w, k2b, _ = self.rhs(w + 0.5 * dt * k1w, b + 0.5 * dt * k1b)
        k3w, k3b, _ = self.rhs(w + 0.5 * dt * k2w, b + 0.5 * dt * k2b)
        k4w, k4b, u1 = self.rhs(w + dt * k3w, b + dt * k3b)
        w = self.filt(w + (dt / 6) * (k1w + 2 * k2w + 2 * k3w + k4w))
        b = self.filt(b + (dt / 6) * (k1b + 2 * k2b + 2 * k3b + k4b))
        return w, b, u1


def run(Nx, Nz, A, stop, device, out):
    eng = BQ2Gpu(Nx, Nz, device)
    dev, f64 = eng.dev, torch.float64
    x = LX * np.arange(Nx) / Nx
    zc = eng.z.cpu().numpy()
    prof = A * (0.5 * (1 - np.cos(x)))[:, None] * np.exp(-30 * (zc / LZ) ** 4)[None, :]
    b = torch.tensor(prof, dtype=f64, device=dev)
    w = torch.zeros((Nx, eng.nz), dtype=f64, device=dev)
    wq = torch.tensor(clencurt(Nz) * (LZ / 2.0), dtype=f64, device=dev)
    dxq = LX / Nx

    def b2i(bf):
        return float(((bf ** 2) * wq[None, :]).sum() * dxq)

    def supgb(bf):
        bh = torch.fft.rfft(bf, dim=0)
        bx = torch.fft.irfft(eng.dxf(bh), n=Nx, dim=0)
        bz = torch.fft.irfft(bh, n=Nx, dim=0) @ eng.Dz.T
        return float(torch.sqrt(bx ** 2 + bz ** 2).max())

    b0 = b2i(b)
    t, it, t0 = 0.0, 0, time.time()
    ser = {"t": [], "sup_gb": [], "b2_drift": []}
    supu = 1.0
    while t < stop and it < 200000:
        dt = min(1e-3, 0.3 * (LX / Nx) / max(supu, 1.0))
        w, b, u1 = eng.step(w, b, dt)
        supu = float(u1.abs().max())
        t += dt; it += 1
        if it % 10 == 0:
            m = supgb(b)
            drift = abs(b2i(b) - b0) / max(b0, 1e-300)
            ser["t"].append(t); ser["sup_gb"].append(m); ser["b2_drift"].append(drift)
            if it % 200 == 0:
                print(f"  t={t:.4f} it={it} sup|grad b|={m:.3e} "
                      f"b2_drift={drift:.2e} dt={dt:.1e} ({time.time()-t0:.0f}s)",
                      flush=True)
            if drift > 3e-3:
                print(f"  b^2 break at t={t:.4f}", flush=True); break
    tt, gg = np.array(ser["t"]), np.array(ser["sup_gb"])
    dd = np.array(ser["b2_drift"]); tr = dd < 1e-3
    res = {"engine": "bq_gpu2", "device": str(dev), "Nx": Nx, "Nz": Nz, "A": A,
           "ceiling": float(gg[tr].max()) if tr.any() else 0.0,
           "t_ceiling": float(tt[tr][np.argmax(gg[tr])]) if tr.any() else 0.0,
           "t_conserved_end": float(tt[tr][-1]) if tr.any() else 0.0,
           "wall_s": round(time.time() - t0, 1), "iters": it,
           "series": {"t": tt.tolist(), "sup_gb": gg.tolist(), "b2_drift": dd.tolist()}}
    pathlib.Path(out).write_text(json.dumps(res, indent=2))
    print(f"[BQ_GPU2] {Nx}x{Nz} A={A:g} {dev}: ceiling {res['ceiling']:.0f} "
          f"conserved to t={res['t_conserved_end']:.3f} | {res['wall_s']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--Nz", type=int, default=256)
    ap.add_argument("--A", type=float, default=4.0)
    ap.add_argument("--stop", type=float, default=2.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="../runs/bq_gpu2.json")
    a = ap.parse_args()
    run(a.Nx, a.Nz, a.A, a.stop, a.device, a.out)
