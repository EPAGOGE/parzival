#!/usr/bin/env python
"""PARZIVAL SWARM -- M1 edition (torch / Apple MPS).

Port of swarm_gpu.py (JAX/CUDA) to Apple Silicon. Same architecture:
  batched dissipative-CLM  w_t = w*Hw - nu*Lambda w  as pure matmul+pointwise,
  slot recycling on fate resolution, bandit mass -> fate-boundary cells,
  confidence ledger with spectral-tail trust flags.

Additions over the CUDA original:
  --model dg      De Gregorio  w_t = -u*w_x + w*Hw - nu*Lambda w  (u_x = Hw),
                  the pre-registered hover-hunt swap (hunches/hover-requires-depletion).
  validation gates (run before science, skip with --skip-gates):
    gate 1: inviscid CLM vs exact solution  w = 4*w0 / ((2 - t*Hw0)^2 + t^2*w0^2)
    gate 2: device fp32 engine vs independent numpy float64 reference
            (fate agreement + blowup-time match on known sub/supercritical amplitudes)
  hover-candidate fate: t > HOVER_T with amplitude neither decayed nor blown
                        (expected ~0 for CLM; the DG prediction is that this turns on)
  vault emitter: appends a hash-chained measurement note to the Obsidian vault.

Run:  ~/gameformer_lab/.venv/bin/python swarm_m1.py            (full test, ~minutes)
      ~/gameformer_lab/.venv/bin/python swarm_m1.py --smoke    (~1 minute)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import time

import numpy as np

BLOWUP_M = 1e3          # sup|w| above this -> blowup fate
DECAY_FRAC = 0.1        # sup|w| below DECAY_FRAC*A0 (after DECAY_TMIN) -> decay fate
DECAY_TMIN = 0.5
HOVER_T = 6.0           # unresolved past this time with moderate amplitude -> hover candidate
HOVER_M_HI = 100.0
DT_MAX = 2e-3
DT_CFL = 0.08
NC = 80                 # bandit cells over amplitude range [3, 8]
A_LO, A_HI = 3.0, 8.0
TAIL_TRUST = 1e-4       # high-k spectral tail fraction above this -> low-trust flag
M1_CORRUPT = 2.0        # step-tripwire threshold, flux-normalized. Calibrated
M2_CORRUPT = 2.0        # 2026-07-23: healthy fp32 envelope over 600 steps x
WEIGHT_EVERY = 300      # 2048 gclm lanes (full life cycle incl. near-blowup)
# maxes at m1=0.19, m2=0.13 -> 10x margin. SENSITIVITY LIMIT, stated plainly:
# this wire catches LANE-SCALE corruption (a 2x stage error fires it; verified
# below); subtle events (1% SDC reads 9.4e-4 -- 63x its cohort but inside the
# global healthy tail) are the fp64/BDF shadow tier's job, not this wire's.
A_STAR_CPU = 5.5348     # ground truth from the CPU bisection run (vault: astar-bisection)


# ---------------------------------------------------------------- operators
def build_mats(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense spectral operator matrices: Hilbert H, |grad| Lambda, d/dx D, u-from-w G.

    H and Lambda are built exactly as in swarm_gpu.py (port fidelity). D and G are
    new (De Gregorio only); D zeroes the Nyquist bin as any odd-order derivative should.
    """
    k = np.fft.rfftfreq(n, 1.0 / n)
    eye = np.eye(n)

    def _h(c: np.ndarray) -> np.ndarray:
        wh = np.fft.rfft(c)
        wh *= -1j * np.sign(k)
        wh[0] = 0
        wh[-1] = 0  # explicit: irfft discards imag Nyquist anyway; don't rely on it
        return np.fft.irfft(wh, n=n)

    def _l(c: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.abs(k) * np.fft.rfft(c), n=n)

    def _d(c: np.ndarray) -> np.ndarray:
        wh = 1j * k * np.fft.rfft(c)
        wh[-1] = 0
        return np.fft.irfft(wh, n=n)

    def _g(c: np.ndarray) -> np.ndarray:
        wh = np.fft.rfft(c)
        out = np.zeros_like(wh)
        out[1:] = -wh[1:] / k[1:]
        out[-1] = 0  # grid-consistent Nyquist velocity is 0 (D kills the bin)
        return np.fft.irfft(out, n=n)

    def mat(op) -> np.ndarray:
        return np.stack([op(eye[:, j]) for j in range(n)], 1)

    return mat(_h), mat(_l), mat(_d), mat(_g)


# ---------------------------------------------------------------- initial conditions
def make_ic(amps: np.ndarray, n: int, ic: str) -> np.ndarray:
    """'cos'  : A*cos(x) -- the CLM standard.
    'cos2' : A*(cos x + 0.3 cos 2x)/1.3, sup-normalized to A.
    NOTE (verified 2026-07-22): every single-mode A*cos(kx) is an EXACT steady
    state of the inviscid De Gregorio nonlinearity, so a --model dg run from
    'cos' decays as A*exp(-nu*t)*cos(x) identically -- zero information.
    DG hunts must use 'cos2' (or richer); CLM keeps 'cos' for comparability."""
    x = 2 * np.pi * np.arange(n) / n
    if ic == "cos":
        return amps[:, None] * np.cos(x)[None, :]
    if ic == "cos2":
        prof = (np.cos(x) + 0.3 * np.cos(2 * x)) / 1.3
        return amps[:, None] * prof[None, :]
    raise ValueError(f"unknown ic {ic!r}")


# ---------------------------------------------------------------- numpy reference engine (float64)
def adv_coeff(model: str, a: float = 1.0) -> float:
    """Advection strength: clm=0, dg=1, gclm=a -- the interpolation dial
    w_t + a*u*w_x = u_x*w - nu*Lambda*w (inviscid transition ~a=0.689)."""
    return {"clm": 0.0, "dg": 1.0}.get(model, a)


def rhs_np(w: np.ndarray, mats: dict, nu: float, model: str,
           a: float = 1.0) -> np.ndarray:
    out = w * (w @ mats["H"].T) - nu * (w @ mats["L"].T)
    adv = adv_coeff(model, a)
    if adv:
        out -= adv * (w @ mats["G"].T) * (w @ mats["D"].T)
    return out


def macro_step_np(w: np.ndarray, t: np.ndarray, mats: dict, nu: float, model: str,
                  a: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    m = np.max(np.abs(w), axis=1)
    dt = np.minimum(DT_MAX, DT_CFL / np.maximum(1.0, m))[:, None]
    k1 = rhs_np(w, mats, nu, model, a)
    k2 = rhs_np(w + 0.5 * dt * k1, mats, nu, model, a)
    k3 = rhs_np(w + 0.5 * dt * k2, mats, nu, model, a)
    k4 = rhs_np(w + dt * k3, mats, nu, model, a)
    return w + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4), t + dt[:, 0]


def resolve_np(amps: np.ndarray, mats: dict, nu: float, model: str, n: int,
               max_steps: int = 30000, ic: str = "cos", a: float = 1.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a small batch to fate in float64. Returns (fates, blowup_times).
    fate: 1 blowup, 0 decay, -1 unresolved."""
    w = make_ic(amps, n, ic)
    t = np.zeros(len(amps))
    fate = np.full(len(amps), -1)
    tstar = np.full(len(amps), np.nan)
    live = np.ones(len(amps), bool)
    for _ in range(max_steps):
        w[live], t[live] = macro_step_np(w[live], t[live], mats, nu, model, a)
        m = np.max(np.abs(w), axis=1)
        newly_blow = live & (m > BLOWUP_M)
        newly_decay = live & (m < DECAY_FRAC * amps) & (t > DECAY_TMIN)
        fate[newly_blow] = 1
        tstar[newly_blow] = t[newly_blow]
        fate[newly_decay] = 0
        live &= ~(newly_blow | newly_decay)
        if not live.any():
            break
    return fate, tstar


# ---------------------------------------------------------------- gates
def gate_exact_inviscid(mats: dict, n: int) -> float:
    """Inviscid CLM, w0 = cos x: exact w(x,t) = 4*w0 / ((2 - t*sin x)^2 + t^2*w0^2).
    Fixed dt so we land exactly on t=0.5. Returns max pointwise error (float64)."""
    x = 2 * np.pi * np.arange(n) / n
    w = np.cos(x)[None, :].copy()
    dt = 1e-3
    for _ in range(500):
        k1 = rhs_np(w, mats, 0.0, "clm")
        k2 = rhs_np(w + 0.5 * dt * k1, mats, 0.0, "clm")
        k3 = rhs_np(w + 0.5 * dt * k2, mats, 0.0, "clm")
        k4 = rhs_np(w + dt * k3, mats, 0.0, "clm")
        w = w + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t = 0.5
    exact = 4 * np.cos(x) / ((2 - t * np.sin(x)) ** 2 + t ** 2 * np.cos(x) ** 2)
    return float(np.max(np.abs(w[0] - exact)))


def gate_cross_backend(engine: "TorchEngine", mats: dict, nu: float, model: str,
                       n: int, ic: str = "cos", a: float = 1.0) -> dict:
    """2a: torch fp64 (CPU) vs numpy fp64 -- implementation equivalence, tight.
    2b: device fp32 vs fp64 -- fates must agree; T* drift is a KNOWN fp32
    characteristic (~7%, rounding noise hastens blowup; measured 2026-07-22),
    logged and tripwired loosely at 15% to catch gross breakage."""
    amps = np.array([4.0, 5.0, 6.5, 7.5])
    fate64, tstar64 = resolve_np(amps, mats, nu, model, n, ic=ic, a=a)
    eng64 = TorchEngine(n, mats, nu, model, "cpu", dtype="float64", a=a)
    fate_t64, tstar_t64 = eng64.resolve(amps, ic=ic)
    fate32, tstar32 = engine.resolve(amps, ic=ic)
    blow = fate64 == 1

    def tdiff(ts):
        return float(np.nanmax(np.abs(tstar64[blow] - ts[blow])
                               / tstar64[blow])) if blow.any() else 0.0

    return {"fates64": fate64.tolist(), "fates_torch64": fate_t64.tolist(),
            "fates32": fate32.tolist(),
            "impl_ok": bool(np.array_equal(fate64, fate_t64)),
            "impl_tstar_rel": tdiff(tstar_t64),
            "fates_ok": bool(np.array_equal(fate64, fate32)),
            "fp32_tstar_rel": tdiff(tstar32),
            "tstar64": np.round(np.nan_to_num(tstar64), 4).tolist()}


# ---------------------------------------------------------------- torch engine
class TorchEngine:
    def __init__(self, n: int, mats: dict, nu: float, model: str, device: str,
                 dtype: str = "float32", a: float = 1.0):
        import torch
        self.torch = torch
        self.n, self.nu, self.model = n, nu, model
        self.adv = adv_coeff(model, a)
        self.dev = torch.device(device)
        self.dt_ = getattr(torch, dtype)
        self.HT = torch.from_numpy(np.ascontiguousarray(mats["H"].T)).to(self.dev, self.dt_)
        self.LT = torch.from_numpy(np.ascontiguousarray(mats["L"].T)).to(self.dev, self.dt_)
        if self.adv:
            self.DT = torch.from_numpy(np.ascontiguousarray(mats["D"].T)).to(self.dev, self.dt_)
            self.GT = torch.from_numpy(np.ascontiguousarray(mats["G"].T)).to(self.dev, self.dt_)

    def rhs(self, w):
        out = w * (w @ self.HT) - self.nu * (w @ self.LT)
        if self.adv:
            out = out - self.adv * (w @ self.GT) * (w @ self.DT)
        return out

    def macro_step(self, w, t, audit: bool = False):
        """audit=True adds Mechanism-2 per-lane hardchecks (vault:
        bsq-solver-ecosystem plan) at the cost of one extra rhs() call:
        M1 (step tripwire): realized energy change vs Simpson quadrature of
          <w,rhs> over the RK4 stages -- a bit-flip/NaN precursor jumps this
          by orders of magnitude (audit.py, measured healthy ~1e-6 at
          dt=2e-3). M2 (aliasing/enstrophy-budget residual): the continuum
          identity <w,rhs(w)> = (1+a/2)*sum(Hw*w^2) - nu*<w,Lw> holds at
          roundoff on smooth/edge states and degrades on under-resolved hot
          states (audit.py, measured 1e-16 / 5.8e-12 / 1.8e-2). Both are
          MONITORS ONLY -- they flag and downgrade trust, never steer or gate
          classification (the probe-is-not-the-loss law)."""
        tc = self.torch
        m = w.abs().amax(dim=1)
        dt = tc.minimum(tc.full_like(m, DT_MAX), DT_CFL / tc.clamp(m, min=1.0)).unsqueeze(1)
        k1 = self.rhs(w)
        mid = w + 0.5 * dt * k1
        k2 = self.rhs(mid)
        mid2 = w + 0.5 * dt * k2
        k3 = self.rhs(mid2)
        k4 = self.rhs(w + dt * k3)
        w1 = w + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not audit:
            return w1, t + dt[:, 0], None, None
        k_end = self.rhs(w1)
        dE = 0.5 * ((w1 * w1).sum(1) - (w * w).sum(1))
        q = (dt[:, 0] / 6) * ((w * k1).sum(1) + 4 * (mid2 * k3).sum(1)
                              + (w1 * k_end).sum(1))
        # Normalize by the energy-FLUX scale, never by |dE|: dE crosses zero
        # at energy turning points, which fp32 lanes hit right before
        # resolving -- a |dE| denominator false-flagged 65% of resolved lanes
        # (measured 2026-07-23). Flux never vanishes on a live lane.
        flux = dt[:, 0] * (w.abs() * k1.abs()).sum(1) + 1e-30
        m1 = (dE - q).abs() / flux
        Hw, u, wx = w @ self.HT, (w @ self.GT if self.adv else w * 0), \
            (w @ self.DT if self.adv else w * 0)
        lhs = (w * (Hw * w - self.nu * (w @ self.LT)
                   - self.adv * u * wx)).sum(1)
        rhs_id = ((1 + self.adv / 2) * (Hw * w * w)).sum(1) \
            - self.nu * (w * (w @ self.LT)).sum(1)
        scale = (Hw.abs() * w * w).sum(1) + 1e-300
        m2 = (lhs - rhs_id).abs() / scale
        return w1, t + dt[:, 0], m1, m2

    def ic(self, amps: np.ndarray, ic: str = "cos"):
        return self.torch.from_numpy(make_ic(amps, self.n, ic)).to(self.dev, self.dt_)

    def resolve(self, amps: np.ndarray, max_steps: int = 30000, ic: str = "cos"
                ) -> tuple[np.ndarray, np.ndarray]:
        """Small-batch fate resolution on device (mirror of resolve_np)."""
        w = self.ic(amps, ic)
        t = self.torch.zeros(len(amps), dtype=self.dt_, device=self.dev)
        fate = np.full(len(amps), -1)
        tstar = np.full(len(amps), np.nan)
        for _ in range(max_steps):
            w, t, _, _ = self.macro_step(w, t)
            m = w.abs().amax(dim=1).cpu().numpy()
            tn = t.cpu().numpy()
            for i in range(len(amps)):
                if fate[i] != -1:
                    continue
                if m[i] > BLOWUP_M:
                    fate[i], tstar[i] = 1, tn[i]
                elif m[i] < DECAY_FRAC * amps[i] and tn[i] > DECAY_TMIN:
                    fate[i] = 0
            if (fate != -1).all():
                break
        return fate, tstar


# ---------------------------------------------------------------- bandit ledger
class Ledger:
    def __init__(self, rng: np.random.Generator, a_lo: float = A_LO,
                 a_hi: float = A_HI):
        self.a_lo, self.a_hi = a_lo, a_hi
        self.edges = np.linspace(a_lo, a_hi, NC + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.n = np.zeros(NC)
        self.blow = np.zeros(NC)
        self.lowtrust = np.zeros(NC)
        self.hover = np.zeros(NC)
        self.corrupt = 0            # Mechanism-2 tripwire: distinct from
        self.wts = np.ones(NC) / NC # lowtrust (resolution) -- this is
        self.rng = rng               # formulation/numerics integrity

    def sample(self, m: int) -> tuple[np.ndarray, np.ndarray]:
        cells = self.rng.choice(NC, size=m, p=self.wts)
        width = self.edges[1] - self.edges[0]
        return self.centers[cells] + width * (self.rng.random(m) - 0.5), cells

    def record(self, cells: np.ndarray, is_blow: np.ndarray,
               is_lowtrust: np.ndarray) -> None:
        """Fate-resolved slots only (blowup/decay). Hover goes to record_hover:
        counting hover in n would deflate p = blow/n exactly in the boundary
        cells the bandit oversamples, biasing Astar_est (verified 2026-07-22)."""
        np.add.at(self.n, cells, 1)
        np.add.at(self.blow, cells, is_blow.astype(float))
        np.add.at(self.lowtrust, cells, is_lowtrust.astype(float))

    def record_hover(self, cells: np.ndarray) -> None:
        np.add.at(self.hover, cells, 1)

    def mark_corrupt(self, count: int) -> None:
        self.corrupt += count

    def reweight(self) -> None:
        nn = np.maximum(self.n, 1)
        p = self.blow / nn
        se = np.sqrt(p * (1 - p) / nn)
        boundary = ((p > 0.02) & (p < 0.98)).astype(float)
        w2 = 0.10 / NC + boundary * (se + 0.02)
        self.wts = w2 / w2.sum()

    def summary(self) -> dict:
        nn = np.maximum(self.n, 1)
        p = self.blow / nn
        band = (p > 0.02) & (p < 0.98) & (self.n >= 10)
        out = {"resolved": int(self.n.sum()),
               "blowups": int(self.blow.sum()),
               "hover_candidates": int(self.hover.sum()),
               "lowtrust": int(self.lowtrust.sum()),
               "corrupt": int(self.corrupt),
               "boundary_cells": np.round(self.centers[band], 4).tolist(),
               "cells": {"centers": np.round(self.centers, 4).tolist(),
                         "n": self.n.astype(int).tolist(),
                         "blow": self.blow.astype(int).tolist(),
                         "hover": self.hover.astype(int).tolist(),
                         "lowtrust": self.lowtrust.astype(int).tolist()}}
        # A* estimate: p crosses 0.5 between adjacent well-sampled cells.
        # Non-monotone p (fp32 flip zone) can produce several crossings: keep
        # the best-supported one and report the count so ambiguity is visible.
        ok = self.n >= 10
        crossings = []
        for i in range(NC - 1):
            if ok[i] and ok[i + 1] and p[i] < 0.5 <= p[i + 1]:
                frac = (0.5 - p[i]) / (p[i + 1] - p[i])
                a = self.centers[i] + frac * (self.centers[i + 1] - self.centers[i])
                crossings.append((min(self.n[i], self.n[i + 1]), a))
        if crossings:
            out["Astar_est"] = round(float(max(crossings)[1]), 4)
            out["n_crossings"] = len(crossings)
        return out


# ---------------------------------------------------------------- vault emitter
def vault_head(vault: pathlib.Path) -> str:
    pat = re.compile(r'attest: \{"prev": "([0-9a-f]+)", "hash": "([0-9a-f]+)"')
    prevs, hashes = set(), set()
    for p in vault.rglob("*.md"):
        m = pat.search(p.read_text())
        if m:
            prevs.add(m.group(1))
            hashes.add(m.group(2))
    heads = hashes - prevs
    if len(heads) != 1:
        raise RuntimeError(f"vault chain has {len(heads)} heads: {heads}")
    return heads.pop()


def emit_note(vault: pathlib.Path, name: str, model_desc: str, trust: str,
              values: dict, links: list[str], body: str = "") -> str:
    prev = vault_head(vault)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    # attest EVERYTHING that carries meaning -- an unhashed trust label would
    # defeat the vault's purpose (flipping quasi->proved must break the chain)
    payload = json.dumps({"name": name, "model": model_desc, "trust": trust,
                          "values": values, "links": links, "body": body,
                          "prev": prev, "ts": ts}, sort_keys=True)
    h = hashlib.sha256(payload.encode()).hexdigest()[:16]
    front = (f'---\ntype: "measurement"\nmodel: "{model_desc}"\ntrust: "{trust}"\n'
             f'values: {json.dumps(values)}\n'
             f'attest: {{"prev": "{prev}", "hash": "{h}", "ts": "{ts}", '
             f'"rule": "sha256(json sort_keys '
             f'{{name,model,trust,values,links,body,prev,ts}})[:16]"}}\n---\n')
    linkline = " ".join(f"[[{l}]]" for l in links)
    text = (f"{front}# {name}\n\nmodel: {model_desc}\ntrust: **{trust}**\n\n"
            f"links: {linkline}\n{body}")
    (vault / "measurements" / f"{name}.md").write_text(text)
    return h


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--nu", type=float, default=1.0)
    ap.add_argument("--model", choices=["clm", "dg", "gclm"], default="clm")
    ap.add_argument("--a", type=float, default=1.0,
                    help="gclm advection dial: 0=CLM, 1=DG (ignored otherwise)")
    ap.add_argument("--ic", choices=["cos", "cos2"], default=None,
                    help="initial-condition family; defaults to cos for clm, "
                         "cos2 otherwise (cos is an exact DG steady state)")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"],
                    default="auto")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 matmul on CUDA Ampere+ (faster, COARSER "
                         "than fp32). OFF by default: TF32 is a distinct meter "
                         "era and its bias must be characterized before use, "
                         "same as the fp32-vs-fp64 shift.")
    ap.add_argument("--search-dtype", choices=["float32", "float64"],
                    default="float32",
                    help="precision of the SEARCH tier. float32 (default): the "
                         "statistical tier -- throughput=precision, near-"
                         "boundary bias measured and fp64-anchored. float64: "
                         "run the search itself in fp64 (removes the ~0.13 "
                         "near-boundary bias + the anchor step; ~2x slower on "
                         "A100/H100, non-starter on a 4090, unavailable on MPS). "
                         "NB: grid error (~0.14 @N128) is comparable, so higher "
                         "--grid attacks accuracy more than fp64 at fixed N.")
    ap.add_argument("--sync-every", type=int, default=1,
                    help="classify fates every K steps instead of every step. "
                         ">1 cuts host<->device round-trips (the CUDA PCIe "
                         "bottleneck); safe for the swarm since T* is unused, "
                         "keep K small so a blown lane can't overflow between "
                         "checks. Audit residuals accumulate as a device-side "
                         "running max so sensitivity is preserved.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", action="store_true",
                    help="Mechanism-2 per-lane M1/M2 hardchecks every step "
                         "(vault: solver-ecosystem plan); flags+recycles "
                         "corrupt lanes, never gates fate classification")
    ap.add_argument("--shadow", type=float, default=0.0,
                    help="Tier-Q redundancy audit: fraction of lanes biopsied "
                         "vs independent fp64 path periodically (catches "
                         "consistent-ALU SDC the in-loop wire misses)")
    ap.add_argument("--alo", type=float, default=A_LO, help="bandit amplitude range low")
    ap.add_argument("--ahi", type=float, default=A_HI, help="bandit amplitude range high")
    ap.add_argument("--smoke", action="store_true", help="quick run: batch 4096, 800 iters")
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--no-vault", action="store_true")
    ap.add_argument("--vault", default=str(pathlib.Path(__file__).parent / "vault"))
    ap.add_argument("--out", default=None, help="write summary JSON here")
    args = ap.parse_args()
    if args.smoke:
        args.batch, args.iters = 4096, 800
    if args.ic is None:
        args.ic = "cos" if args.model == "clm" else "cos2"

    try:
        import torch
    except ImportError:
        raise SystemExit("torch not found -- run with ~/gameformer_lab/.venv/bin/python")
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    if args.device == "cuda":
        # discipline: default to true fp32, not TF32 -- a silent precision
        # change is a silent meter-era change (cf the fp32-vs-fp64 boundary
        # shift). Opt in explicitly with --tf32 once its bias is measured.
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    if args.search_dtype == "float64" and args.device == "mps":
        raise SystemExit("MPS has no fp64; use --device cpu or cuda for "
                         "--search-dtype float64")
    n = args.grid
    Hm, Lm, Dm, Gm = build_mats(n)
    mats = {"H": Hm, "L": Lm, "D": Dm, "G": Gm}
    engine = TorchEngine(n, mats, args.nu, args.model, args.device, a=args.a,
                         dtype=args.search_dtype)
    print(f"parzival swarm_m1 | model={args.model} adv={engine.adv} nu={args.nu} "
          f"N={n} B={args.batch} iters={args.iters} device={args.device}")

    gates = {}
    if not args.skip_gates:
        t0 = time.time()
        err = gate_exact_inviscid(mats, n)
        gates["exact_inviscid_err"] = err
        assert err < 1e-6, f"GATE 1 FAIL: inviscid exact-solution error {err:.3e}"
        print(f"gate 1 PASS  inviscid vs exact: max err {err:.3e}")
        xb = gate_cross_backend(engine, mats, args.nu, args.model, n, ic=args.ic,
                                a=args.a)
        gates["cross_backend"] = xb
        assert xb["impl_ok"] and xb["impl_tstar_rel"] < 1e-3, \
            f"GATE 2a FAIL: torch port not equivalent to numpy reference {xb}"
        assert xb["fates_ok"], f"GATE 2b FAIL: fp32 fate mismatch {xb}"
        assert xb["fp32_tstar_rel"] < 0.15, f"GATE 2b FAIL: fp32 T* drift gross {xb}"
        print(f"gate 2a PASS  torch fp64 == numpy fp64 (T* rel "
              f"{xb['impl_tstar_rel']:.1e})")
        print(f"gate 2b PASS  fp32/{args.device} fates {xb['fates32']} match; "
              f"T* fp32 drift {xb['fp32_tstar_rel']:.3f} (known fp32 "
              f"characteristic, ~0.07) ({time.time()-t0:.1f}s)")

    # ---- swarm
    rng = np.random.default_rng(args.seed)
    led = Ledger(rng, args.alo, args.ahi)
    amps, cell = led.sample(args.batch)
    a0 = amps.copy()
    w = engine.ic(amps, args.ic)
    t = torch.zeros(args.batch, device=engine.dev)
    t0 = time.time()
    steps = 0
    dead_total = 0
    sync_every = max(1, args.sync_every)
    m1acc = m2acc = None                     # device-side running-max residuals
    last_reweight = 0                        # first reweight ~it=WEIGHT_EVERY
    last_shadow = -(WEIGHT_EVERY // 2)       # offset shadow to mid-period
    for it in range(args.iters):
        w, t, m1r, m2r = engine.macro_step(w, t, audit=args.audit)
        steps += args.batch
        if args.audit:                       # accumulate max across the window
            m1acc = m1r if m1acc is None else torch.maximum(m1acc, m1r)
            m2acc = m2r if m2acc is None else torch.maximum(m2acc, m2r)
        # only touch the host every sync_every steps (CUDA PCIe economy) --
        # and always on the final iter so nothing is left unresolved
        if (it % sync_every != sync_every - 1) and (it != args.iters - 1):
            continue
        m = w.abs().amax(dim=1).cpu().numpy()
        tn = t.cpu().numpy()
        is_corrupt = np.zeros(args.batch, bool)
        if args.audit:
            is_corrupt = ((m1acc > M1_CORRUPT) | (m2acc > M2_CORRUPT)).cpu().numpy()
            m1acc = m2acc = None             # reset window
            if is_corrupt.any():
                led.mark_corrupt(int(is_corrupt.sum()))
                idx = torch.from_numpy(np.where(is_corrupt)[0]).to(engine.dev)
                anew, cnew = led.sample(int(is_corrupt.sum()))
                w[idx] = engine.ic(anew, args.ic)
                t[idx] = 0.0
                a0[is_corrupt] = anew
                cell[is_corrupt] = cnew
        is_dead = np.isnan(m) & ~is_corrupt  # poisoned lane: recycle, never ledger
        is_blow = (m > BLOWUP_M) & ~is_corrupt  # inf counts as blowup; NaN>x False
        is_decay = (~is_blow) & (m < DECAY_FRAC * a0) & (tn > DECAY_TMIN) & ~is_corrupt
        is_hover = (~is_blow) & (~is_decay) & (~is_dead) & ~is_corrupt & \
                   (tn > HOVER_T) & (m < HOVER_M_HI)
        fated = np.where(is_blow | is_decay)[0]
        hov = np.where(is_hover)[0]
        dead = np.where(is_dead)[0]
        if len(fated):
            wd = w[torch.from_numpy(fated).to(engine.dev)].cpu().numpy()
            sp = np.abs(np.fft.rfft(wd, axis=1))
            frac = (sp[:, 3 * sp.shape[1] // 4:] ** 2).sum(1) / np.maximum(
                (sp ** 2).sum(1), 1e-30)
            # non-finite spectrum (inf blowup caught late) is the LEAST trusted
            led.record(cell[fated], is_blow[fated],
                       (frac > TAIL_TRUST) | ~np.isfinite(frac))
        if len(hov):
            led.record_hover(cell[hov])
        dead_total += len(dead)
        done = np.concatenate([fated, hov, dead])
        if len(done):
            anew, cnew = led.sample(len(done))
            idx = torch.from_numpy(done).to(engine.dev)
            w[idx] = engine.ic(anew, args.ic)
            t[idx] = 0.0
            a0[done] = anew
            cell[done] = cnew
        # periodic maintenance: due-counters (not it%WEIGHT_EVERY equality) so
        # a sync_every stride can't skip the exact trigger iteration
        if args.shadow > 0 and it - last_shadow >= WEIGHT_EVERY:
            last_shadow = it
            import shadow as _sh                     # lazy: shadow imports us
            si = rng.choice(args.batch, size=max(16, int(args.shadow * args.batch)),
                            replace=False)
            sr = _sh.shadow_audit(engine, w[torch.from_numpy(si).to(engine.dev)],
                                  a0[si], mats, args.model, args.a)
            flagged = si[sr["flags"]]
            if len(flagged) or sr["systematic_fault"]:
                led.mark_corrupt(len(flagged))
                if len(flagged):
                    idx = torch.from_numpy(flagged).to(engine.dev)
                    anew, cnew = led.sample(len(flagged))
                    w[idx] = engine.ic(anew, args.ic)
                    t[idx] = 0.0
                    a0[flagged] = anew
                    cell[flagged] = cnew
                print(f"[shadow] it={it+1} biopsied {sr['n']} flagged "
                      f"{sr['n_flagged']} systematic={sr['systematic_fault']}")
        if it - last_reweight >= WEIGHT_EVERY:
            last_reweight = it
            led.reweight()
            print(f"it={it+1} resolved={int(led.n.sum())} "
                  f"sps={steps/(time.time()-t0):,.0f}")

    elapsed = time.time() - t0
    summ = led.summary()
    summ.update({"sps": int(steps / elapsed), "elapsed_s": round(elapsed, 1),
                 "batch": args.batch, "iters": args.iters, "device": args.device,
                 "model": args.model, "adv": adv_coeff(args.model, args.a),
                 "nu": args.nu, "ic": args.ic,
                 "arange": [args.alo, args.ahi],
                 # meter-era provenance: precision path + hardcheck settings
                 "meter": {"tf32": bool(args.tf32), "sync_every": sync_every,
                           "search_dtype": args.search_dtype,
                           "audit": bool(args.audit), "shadow": args.shadow},
                 "dead_lanes": dead_total, "gates": gates})

    # fp64 anchor: the swarm's fates are device-fp32; re-resolve the boundary
    # band (+/- one cell, extending upward if fp64 finds no blowup there) with
    # the float64 reference so the vaulted note carries a precision-honest bracket.
    if summ["boundary_cells"]:
        width = (args.ahi - args.alo) / NC
        probe = sorted({round(a, 4) for c in summ["boundary_cells"]
                        for a in (c - width, c, c + width)
                        if args.alo <= a <= args.ahi})[:8]
        f64, _ = resolve_np(np.array(probe), mats, args.nu, args.model, n,
                            max_steps=120000, ic=args.ic, a=args.a)
        while 1 not in f64 and probe[-1] + width <= args.ahi and len(probe) < 14:
            nxt = round(probe[-1] + width, 4)
            fx, _ = resolve_np(np.array([nxt]), mats, args.nu, args.model, n,
                               max_steps=120000, ic=args.ic, a=args.a)
            probe.append(nxt)
            f64 = np.append(f64, fx[0])
        summ["fp64_check"] = {"amps": probe, "fates": f64.tolist()}
        dec = [a for a, f in zip(probe, f64) if f == 0]
        blo = [a for a, f in zip(probe, f64) if f == 1]
        if dec and blo:
            summ["Astar_fp64_bracket"] = [max(dec), min(blo)]

    print(json.dumps({k: v for k, v in summ.items() if k != "gates"}, indent=2))
    if "Astar_est" in summ:
        print(f"A* fp32-swarm estimate {summ['Astar_est']} | fp64 bracket "
              f"{summ.get('Astar_fp64_bracket', 'n/a')} | vault bisection "
              f"{A_STAR_CPU} (grid-dependent, see astar-grid-systematic)")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summ, indent=2))
    if not args.no_vault:
        stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
        vals = {k: summ[k] for k in ("Astar_est", "Astar_fp64_bracket", "resolved",
                                     "blowups", "hover_candidates", "dead_lanes",
                                     "corrupt", "sps", "batch", "iters", "device",
                                     "model", "adv", "ic", "arange", "meter")
                if k in summ}
        try:
            tag = f"-a{args.a:g}" if args.model == "gclm" else ""
            h = emit_note(pathlib.Path(args.vault),
                          f"swarm-m1-{args.model}{tag}-{stamp}",
                          f"swarm engine M1/{args.device} B={args.batch}", "quasi",
                          vals, ["swarm-engine", "critical-threshold"],
                          body=f"\nboundary cells: {summ['boundary_cells']}\n"
                               f"fp64 check: {json.dumps(summ.get('fp64_check', {}))}\n"
                               f"gates: {json.dumps(gates.get('cross_backend', {}))}\n")
            print(f"vault note appended, chain head now {h}")
        except Exception as exc:  # vault trouble must not destroy run outputs
            print(f"VAULT EMIT FAILED (outputs already saved): {exc}")


if __name__ == "__main__":
    main()
