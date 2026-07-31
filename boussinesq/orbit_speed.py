#!/usr/bin/env python3
"""Orbit speed V in profile space. S1-compliant: THE script for these numbers.
V = ||dU|| / (||U|| dlnA): profile change per e-folding. Settling: V decays.
Wandering: V stays O(1). Anchor: SYMa/SYMb exact symmetry, cross-diff ~ 0.
FIXED 2026-07-30: extent gate is TOTAL HWHM per axis (cz1+cz2>=6), never the
min -- the peak sits cells from the wall, so the one-sided extent is always
small and a min-gate silently culls every snapshot (the ValueError crash)."""
import glob, h5py, json, numpy as np, sys
sys.path.insert(0, '.')
from lambda_geom import axes, vorticity

def hw(prof, i0):
    pk = prof[i0]; h = pk/2; n = len(prof); j = i0
    while j-i0 < n-1 and prof[(j+1) % n] > h: j += 1
    k = i0
    while i0-k < n-1 and prof[(k-1) % n] > h: k -= 1
    return i0-k, j-i0

def trust_window(tag):
    """Longest stretch with spectral tail <=1e-6 AND gamma_drift <=1e-4.
    ADDED 2026-07-30 after a pre-mortem: this module had NO gate. G5_11 was
    clean for 52/188 records (gamma_drift peaked at 2.0 -- sup|r^2 u1| is
    exactly conserved and CANNOT increase, so 2.0 is total numerical failure)
    and the orbit slope was being fitted straight through the broken part."""
    try:
        r=[json.loads(l) for l in open(f"../runs/stream_{tag}.jsonl") if l.strip()]
    except Exception:
        return (0.0, float("inf"))
    t=np.array([x["t"] for x in r])
    ok=np.array([max(x.get("tail_u1",0),x.get("tail_w1",0))<=1e-6
                 and x.get("gamma_drift",0)<=1e-4 for x in r])
    best=(0,0); i=0
    while i<len(ok):
        if ok[i]:
            j=i
            while j<len(ok) and ok[j]: j+=1
            if j-i>best[1]-best[0]: best=(i,j)
            i=j
        else: i+=1
    return (t[best[0]], t[best[1]-1]) if best[1]>best[0] else (0.0, 0.0)


def profiles(tag, mode="iso"):
    """mode='aniso': rescale z,r by their OWN HWHM (divides the aspect ratio
    OUT -- blind to a drifting second length scale, i.e. blind to exactly the
    wandering signature). mode='iso': rescale BOTH axes by one length
    (geometric mean of the HWHMs), so aspect drift stays IN the profile and
    is counted as orbit motion. 'iso' is the correct observable for the
    settling-vs-wandering question; 'aniso' is kept to expose the difference.
    Also returns the peak location for a structure-identity gate: if the peak
    jumps, consecutive profiles are DIFFERENT OBJECTS and the pair is void."""
    out = []
    t0, t1 = trust_window(tag)
    for fn in sorted(glob.glob(f"../runs/snap_{tag}/*.h5")):
        with h5py.File(fn, 'r') as f:
            z, r = axes(f)
            W, U, st = f["tasks"]["omega1"][:], f["tasks"]["u1"][:], f["scales/sim_time"][:]
            for n in range(len(st)):
                if st[n] <= 0 or st[n] < t0 or st[n] > t1: continue
                wr, wt, wz = vorticity(U[n], W[n], z, r)
                mag = np.sqrt(wr**2 + wt**2 + wz**2); A = mag.max()
                iz, ir = np.unravel_index(np.argmax(mag), mag.shape)
                cz1, cz2 = hw(mag[:, ir], iz); cr1, cr2 = hw(mag[iz, :], ir)
                if cz1+cz2 < 6 or cr1+cr2 < 6: continue        # TOTAL extent
                ys = np.linspace(-2.0, 2.0, 25)
                hz = max((cz1+cz2)//2, 3); hr = max((cr1+cr2)//2, 3)
                if mode == "iso":
                    # one length for both axes in GRID-CELL units, converted per
                    # axis by the local spacing so it is a genuine single
                    # physical length; aspect drift then shows up as shape change
                    dz = float(np.median(np.abs(np.diff(z))))
                    dr = float(np.median(np.abs(np.diff(r))))
                    Lphys = float(np.sqrt((hz*dz)*(hr*dr)))
                    hz = max(int(round(Lphys/dz)), 3); hr = max(int(round(Lphys/dr)), 3)
                zi = (iz + np.round(ys*hz).astype(int)) % len(z)
                ri = ir + np.round(ys*hr).astype(int)
                ok = (ri >= 0) & (ri < len(r))
                out.append((float(st[n]), float(A), mag[np.ix_(zi, ri[ok])]/A,
                            float(z[iz]), float(r[ir])))
    return out

def vseq(P, jump_tol=0.02):
    """Consecutive-pair orbit speed, with a STRUCTURE-IDENTITY gate: if the
    peak location moves more than jump_tol between snapshots the two profiles
    belong to different structures and the pair is discarded. Without this,
    a wandering argmax is measured as profile motion."""
    v = []; voided = 0
    for a, b in zip(P, P[1:]):
        dg = np.log(b[1]/a[1])
        if abs(dg) < 1e-3: continue
        if np.hypot(b[3]-a[3], b[4]-a[4]) > jump_tol:
            voided += 1; continue
        m = min(a[2].shape[1], b[2].shape[1])
        v.append((np.log(b[1]),
                  float(np.linalg.norm(b[2][:, :m]-a[2][:, :m]) /
                        np.linalg.norm(a[2][:, :m]) / abs(dg))))
    return v, voided

def verdict(v, nboot=2000, rng=None):
    """Post-transient slope with a bootstrap CI.

    ASSEMBLY TRANSIENT: omega1 starts at exactly 0, so the first snapshots
    measure the structure being BORN, not an orbit. That phase shows V ~ 30
    and decaying steeply; fitting through it returns a spurious 'SETTLING'
    (measured: -3.99 on GEN42). Drop leading points until V falls below
    3x the median of the back half, and report how many were dropped.

    A slope is only reported as a verdict when its 90% bootstrap CI lies
    entirely inside one regime. Otherwise INCONCLUSIVE, with the extra
    e-foldings needed to separate them."""
    import numpy as _np
    rng = _np.random.default_rng(0) if rng is None else rng
    lnA = _np.array([x[0] for x in v]); V = _np.array([x[1] for x in v])
    med = _np.median(V[len(V)//2:])
    k = 0
    while k < len(V)-4 and V[k] > 3*med: k += 1
    lnA, V = lnA[k:], V[k:]
    if len(V) < 6:
        return dict(n=len(V), dropped=k, slope=None, lo=None, hi=None,
                    span=0.0, label="TOO FEW post-transient points")
    x = lnA - lnA[0]; y = _np.log(_np.maximum(V, 1e-9))
    sl = float(_np.polyfit(x, y, 1)[0])
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(x), len(x))
        if len(set(i.tolist())) < 3: continue
        bs.append(_np.polyfit(x[i], y[i], 1)[0])
    lo, hi = (float(_np.percentile(bs, 5)), float(_np.percentile(bs, 95))) if bs else (sl, sl)
    if hi < -0.15:   lab = "SETTLING"
    elif lo > 0.15:  lab = "DEPARTING"
    elif lo >= -0.15 and hi <= 0.15: lab = "WANDERING (O(1) per e-fold)"
    else:            lab = "INCONCLUSIVE"
    return dict(n=len(V), dropped=k, slope=sl, lo=lo, hi=hi,
                span=float(x[-1]), label=lab)


def main(tags):
    Pa, Pb = profiles("SYMa"), profiles("SYMb")
    dev = []
    if Pa and Pb:
        ga = np.array([x[1]/Pa[0][1] for x in Pa])
        gb = np.array([x[1]/Pb[0][1] for x in Pb])
        for i in range(len(Pa)):
            j = int(np.argmin(np.abs(np.log(gb/ga[i]))))
            if abs(np.log(gb[j]/ga[i])) > 0.10: continue
            m = min(Pa[i][2].shape[1], Pb[j][2].shape[1])
            dev.append(float(np.linalg.norm(Pa[i][2][:, :m]-Pb[j][2][:, :m]) /
                             np.linalg.norm(Pa[i][2][:, :m])))
            _ = None
    if dev:
        print(f"ANCHOR exact-symmetry noise floor: mean {np.mean(dev):.4f} "
              f"max {np.max(dev):.4f} n={len(dev)}")
    else:
        print("ANCHOR: no matched pairs (SYM runs missing)")
    for tag in tags:
        P = profiles(tag, mode="iso")
        growth = (P[-1][1]/P[0][1]) if len(P) > 1 else 0.0
        v, voided = vseq(P)
        va, _ = vseq(profiles(tag, mode="aniso"))
        if len(v) < 6:
            print(f"{tag}: too few usable pairs ({len(v)}, {voided} voided by peak jump)")
            continue
        d = verdict(v); da = verdict(va) if len(va) >= 6 else None
        if d["slope"] is None:
            print(f"{tag:<10} {d['label']}  (n={d['n']}, dropped {d['dropped']} transient)")
            continue
        print(f"{tag:<10} slope {d['slope']:+.3f}  90% CI [{d['lo']:+.3f},{d['hi']:+.3f}]  "
              f"n={d['n']} (-{d['dropped']} transient)  span {d['span']:.2f} e-folds  "
              f"-> {d['label']}")
        print(f"           amplitude growth x{growth:.1f}  |  {voided} pairs voided "
              f"(peak jump)  |  aniso-rescaled slope "
              f"{('%+.3f' % da['slope']) if da and da['slope'] is not None else 'n/a'}"
              f"  <- the blind variant")
        if d["label"] == "INCONCLUSIVE":
            w = d["hi"] - d["lo"]
            need = d["span"] * ((w / 0.30) ** 2 - 1)
            print(f"           CI width {w:.3f} vs 0.30 needed; "
                  f"~{max(need,0):.1f} more e-folds of growth would separate the regimes")

if __name__ == "__main__":
    main(sys.argv[1:] or ["OR_z256r768", "NUL1e-4", "GEN42"])
