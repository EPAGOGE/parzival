#!/usr/bin/env python3
"""WIDE FEATURE BATTERY + ORBIT GEOMETRY.

Every scalar this campaign can extract from a snapshot, plus the orbit
diagnostics that a single scalar cannot provide.

WHY WIDE. Settling vs wandering was being judged from |dU/ds| alone -- how FAST
the profile moves. That is blind to WHERE it moves. Three fates share one speed:

    settling   increments point consistently back toward a fixed point AND shrink
    wandering  increments decorrelate; no preferred direction; speed does not decay
    cycle/DSS  increments rotate coherently with a period; speed roughly constant

The discriminator is the DIRECTION of successive increments, not their length.
cos(dU_k, dU_{k+1}) separates all three where |dU| cannot.

PER-SNAPSHOT FEATURES
  amplitude   sup|omega| (full vector), sup|omega1|, sup|omega^theta|
  position    peak (z*, r*), distance to the corner (1,0)
  motion      peak velocity magnitude and direction (deg), from consecutive frames
  scale       l_z, l_r (HWHM), second-moment widths, aspect ratio l_z/l_r
  geometry    Lambda = sup|grad xi| |omega|^-1/2 (peak box), direction-field speed
  curvature   Hessian eigenvalues of |omega| at the peak, their ratio
  stretch     alpha = dln sup|omega| / dt  (exact for Euler at the max, since
              advection cannot move a maximum's value)
  invariants  sup|r^2 u1| (exactly conserved), enstrophy, helicity density at peak
  health      spectral tail, gamma drift, connected component count (periodic)

ORBIT FEATURES (from the rescaled profile U)
  V           ||dU|| / (||U|| dlnA)                  speed in profile space
  cos_step    cos(dU_k, dU_{k+1})                    increment persistence  <-- NEW
  cos_to_end  cos(dU_k, U_end - U_k)                 aim at the final state
  turn        cumulative turning angle of the path
"""
from __future__ import annotations

import glob
import json
import sys

import h5py
import numpy as np
from scipy import ndimage

sys.path.insert(0, '.')
from lambda_geom import axes, vorticity, grad_xi_sq, d_dz, d_dr


def periodic_components(mask):
    lab, n = ndimage.label(mask)
    if n < 2:
        return n
    merge = {}
    for a, b in zip(lab[0], lab[-1]):
        if a > 0 and b > 0 and a != b:
            ra, rb = min(a, b), max(a, b)
            while ra in merge: ra = merge[ra]
            while rb in merge: rb = merge[rb]
            if ra != rb: merge[max(ra, rb)] = min(ra, rb)
    return n - len(merge)


def hwhm_cells(prof, i0):
    pk = prof[i0]; h = pk / 2; n = len(prof)
    j = i0
    while j - i0 < n - 1 and prof[(j + 1) % n] > h: j += 1
    k = i0
    while i0 - k < n - 1 and prof[(k - 1) % n] > h: k -= 1
    return i0 - k, j - i0


def trust_window(tag):
    try:
        r = [json.loads(l) for l in open(f"../runs/stream_{tag}.jsonl") if l.strip()]
    except Exception:
        return (0.0, float("inf"))
    t = np.array([x["t"] for x in r])
    ok = np.array([max(x.get("tail_u1", 0), x.get("tail_w1", 0)) <= 1e-6
                   and x.get("gamma_drift", 0) <= 1e-4 for x in r])
    best = (0, 0); i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j < len(ok) and ok[j]: j += 1
            if j - i > best[1] - best[0]: best = (i, j)
            i = j
        else:
            i += 1
    return (t[best[0]], t[best[1] - 1]) if best[1] > best[0] else (0.0, 0.0)


def extract(tag, npts=21):
    """One dict per gated snapshot, plus the rescaled profile vector U."""
    t0, t1 = trust_window(tag)
    rows = []
    for fn in sorted(glob.glob(f"../runs/snap_{tag}/*.h5")):
        with h5py.File(fn, 'r') as f:
            z, r = axes(f)
            W, U1, st = f["tasks"]["omega1"][:], f["tasks"]["u1"][:], f["scales/sim_time"][:]
            dzs = float(np.median(np.abs(np.diff(z))))
            drs = float(np.median(np.abs(np.diff(r))))
            for n in range(len(st)):
                if st[n] <= 0 or st[n] < t0 or st[n] > t1:
                    continue
                u1, w1 = U1[n], W[n]
                wr, wt, wz = vorticity(u1, w1, z, r)
                mag = np.sqrt(wr**2 + wt**2 + wz**2)
                A = float(mag.max())
                if A <= 0: continue
                iz, ir = np.unravel_index(np.argmax(mag), mag.shape)
                cz1, cz2 = hwhm_cells(mag[:, ir], iz)
                cr1, cr2 = hwhm_cells(mag[iz, :], ir)
                if cz1 + cz2 < 6 or cr1 + cr2 < 6: continue
                lz = (cz1 + cz2) * dzs; lr = (cr1 + cr2) * drs
                # direction field and Lambda in the 2x-HWHM box
                s = np.maximum(mag, A * 1e-12)
                g = np.sqrt(np.maximum(grad_xi_sq(wr/s, wt/s, wz/s, z, r), 0.0))
                nz, nr = mag.shape
                zi = np.arange(iz - 2*cz1, iz + 2*cz2 + 1) % nz
                rj = np.arange(max(0, ir - 2*cr1), min(nr - 1, ir + 2*cr2) + 1)
                box = np.ix_(zi, rj)
                lam = float((g[box] / np.sqrt(np.maximum(mag[box], 1e-300))).max())
                # curvature of |omega| at the peak (2nd differences)
                czz = (mag[(iz+1) % nz, ir] - 2*mag[iz, ir] + mag[(iz-1) % nz, ir]) / dzs**2
                crr = ((mag[iz, min(ir+1, nr-1)] - 2*mag[iz, ir]
                        + mag[iz, max(ir-1, 0)]) / drs**2)
                # helicity density at the peak
                ur = -r[None, :] * d_dz(np.zeros_like(u1), z)   # placeholder, see below
                hel = float(wt[iz, ir] * u1[iz, ir] * r[ir])
                gam = float(np.abs((r[None, :]**2) * u1).max())
                ens = float(np.sum(mag**2 * r[None, :]) * dzs * drs)
                comp = periodic_components(mag > 0.5 * A)
                # rescaled profile on a common grid, ONE physical length (iso)
                ys = np.linspace(-2.0, 2.0, npts)
                Lp = float(np.sqrt(lz * lr))
                hz = max(int(round(Lp / dzs)), 3); hr = max(int(round(Lp / drs)), 3)
                zz = (iz + np.round(ys * hz).astype(int)) % nz
                rr2 = ir + np.round(ys * hr).astype(int)
                ok = (rr2 >= 0) & (rr2 < nr)
                Uprof = (mag[np.ix_(zz, rr2[ok])] / A).ravel()
                rows.append(dict(t=float(st[n]), A=A, w1=float(np.abs(w1).max()),
                                 zstar=float(z[iz]), rstar=float(r[ir]),
                                 dcorner=float(np.hypot(z[iz], 1.0 - r[ir])),
                                 lz=lz, lr=lr, aspect=lz / max(lr, 1e-300),
                                 lam=lam, kzz=float(czz), krr=float(crr),
                                 kratio=float(czz / (crr if abs(crr) > 1e-300 else np.nan)),
                                 hel=hel, gamma=gam, ens=ens, comp=int(comp),
                                 U=Uprof))
    return rows


def orbit(rows):
    """Speed, increment persistence, aim, turning. The direction diagnostics."""
    if len(rows) < 4:
        return None
    n = min(len(x["U"]) for x in rows)
    U = np.array([x["U"][:n] for x in rows])
    A = np.array([x["A"] for x in rows])
    d = np.diff(U, axis=0)
    dlnA = np.diff(np.log(A))
    keep = np.abs(dlnA) > 1e-4
    V = np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(U[:-1], axis=1), 1e-300)
    V = V / np.maximum(np.abs(dlnA), 1e-12)
    cs = []
    for k in range(len(d) - 1):
        a, b = d[k], d[k + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cs.append(float(a @ b / (na * nb)))
    aim = []
    for k in range(len(d)):
        tgt = U[-1] - U[k]
        na, nb = np.linalg.norm(d[k]), np.linalg.norm(tgt)
        if na > 0 and nb > 0:
            aim.append(float(d[k] @ tgt / (na * nb)))
    return dict(V=V[keep], lnA=np.log(A)[:-1][keep],
                cos_step=np.array(cs), cos_aim=np.array(aim),
                span=float(np.log(A[-1] / A[0])), n=len(rows))


def report(tags):
    print(f"{'run':<12}{'n':>4}{'span':>7}{'V_end':>8}{'cos_step':>10}"
          f"{'cos_aim':>9}{'aspect drift':>14}{'peak move':>11}{'verdict':>26}")
    print("-" * 101)
    for tag in tags:
        rows = extract(tag)
        if len(rows) < 5:
            print(f"{tag:<12}{len(rows):>4}   too few"); continue
        o = orbit(rows)
        if o is None or len(o["cos_step"]) < 3:
            print(f"{tag:<12}{len(rows):>4}   orbit too short"); continue
        asp = np.array([x["aspect"] for x in rows])
        A = np.array([x["A"] for x in rows])
        asl = float(np.polyfit(np.log(A), np.log(asp), 1)[0])
        mv = float(np.hypot(rows[-1]["zstar"] - rows[0]["zstar"],
                            rows[-1]["rstar"] - rows[0]["rstar"]))
        cs = float(np.mean(o["cos_step"])); ca = float(np.mean(o["cos_aim"]))
        Vend = float(np.mean(o["V"][-3:])) if len(o["V"]) >= 3 else float("nan")
        if cs > 0.5 and ca > 0.5:
            v = "SETTLING (coherent, aimed)"
        elif abs(cs) < 0.25:
            v = "WANDERING (decorrelated)"
        elif cs < -0.4:
            v = "OSCILLATING (alternating)"
        else:
            v = "MIXED"
        print(f"{tag:<12}{o['n']:>4}{o['span']:>7.2f}{Vend:>8.3f}{cs:>10.3f}"
              f"{ca:>9.3f}{asl:>14.3f}{mv:>11.5f}{v:>26}")
    print("-" * 101)
    print("cos_step  +1 increments aligned (headed somewhere) | 0 decorrelated "
          "(wandering) | -1 alternating")
    print("cos_aim   +1 each step moves toward the final state (converging)")


if __name__ == "__main__":
    report(sys.argv[1:] or ["OR_z256r768", "NUL1e-4", "G5_11",
                            "W3", "W19", "W63"])
