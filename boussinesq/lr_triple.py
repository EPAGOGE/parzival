#!/usr/bin/env python3
"""Is the 8.9x IC-to-IC spread in the RADIAL collapse rate physical, or is it
variance in a hard-to-measure quantity dressed as physics?

Three ESTIMATORS of the radial scale, sharing only the peak location:
  M1 HWHM        half-width at half-max along r through the peak  (original)
  M2 curvature   l = sqrt(2 |omega| / |d2omega/dr2|) at the peak: a purely LOCAL
                 second-derivative scale, immune to the far tail and to the
                 one-sided wall truncation that M1 suffers
  M3 integral    l = (integral of |omega| dr) / |omega|_peak along the peak row:
                 a purely GLOBAL first-moment scale, immune to the local curvature
                 noise M2 suffers

M2 and M3 fail in opposite ways, so agreement between all three is strong.
The DECISIVE test is not the value of any slope but whether the IC-to-IC
SPREAD survives. Internal control: l_z measured the same three ways must stay
tight (sd ~ 0.08), since that is the quantity already shown to be universal.
"""
import sys, numpy as np
sys.path.insert(0, '.')
import glob, h5py
from features import trust_window, hwhm_cells
from lambda_geom import axes, vorticity

def scales(tag):
    t0, t1 = trust_window(tag); out = []
    for fn in sorted(glob.glob(f"../runs/snap_{tag}/*.h5")):
        with h5py.File(fn, 'r') as f:
            z, r = axes(f)
            W, U, st = f["tasks"]["omega1"][:], f["tasks"]["u1"][:], f["scales/sim_time"][:]
            dzs = float(np.median(np.abs(np.diff(z)))); drs = float(np.median(np.abs(np.diff(r))))
            for n in range(len(st)):
                if st[n] <= 0 or st[n] < t0 or st[n] > t1: continue
                wr, wt, wz = vorticity(U[n], W[n], z, r)
                mag = np.sqrt(wr**2 + wt**2 + wz**2); A = float(mag.max())
                if A <= 0: continue
                iz, ir = np.unravel_index(np.argmax(mag), mag.shape)
                cz1, cz2 = hwhm_cells(mag[:, ir], iz); cr1, cr2 = hwhm_cells(mag[iz, :], ir)
                if cz1+cz2 < 6 or cr1+cr2 < 6: continue
                nz, nr = mag.shape
                rowr = mag[iz, :]; rowz = mag[:, ir]
                # M1 HWHM
                lr1 = (cr1+cr2)*drs; lz1 = (cz1+cz2)*dzs
                # M2 local curvature
                d2r = (rowr[min(ir+1, nr-1)] - 2*rowr[ir] + rowr[max(ir-1, 0)])/drs**2
                d2z = (rowz[(iz+1) % nz] - 2*rowz[iz] + rowz[(iz-1) % nz])/dzs**2
                lr2 = np.sqrt(2*A/abs(d2r)) if abs(d2r) > 0 else np.nan
                lz2 = np.sqrt(2*A/abs(d2z)) if abs(d2z) > 0 else np.nan
                # M3 integral width
                lr3 = float(rowr.sum()*drs/A); lz3 = float(rowz.sum()*dzs/A)
                out.append((A, lr1, lr2, lr3, lz1, lz2, lz3))
    return np.array(out)

def slopes(tag):
    a = scales(tag)
    if len(a) < 8: return None
    x = np.log(a[:, 0])
    s = []
    for c in range(1, 7):
        y = a[:, c]
        ok = np.isfinite(y) & (y > 0)
        s.append(float(np.polyfit(x[ok], np.log(y[ok]), 1)[0]) if ok.sum() > 5 else np.nan)
    return s

RUNS = ["OR_z256r768", "NUL1e-4", "W3", "W11", "W19", "W23", "W42", "W51", "W63"]
print(f"{'run':<13}{'lr HWHM':>9}{'lr curv':>9}{'lr integ':>10}  |{'lz HWHM':>9}{'lz curv':>9}{'lz integ':>10}")
print("-"*72)
R = {}
for t in RUNS:
    s = slopes(t)
    if s is None: print(f"{t:<13}  too few"); continue
    R[t] = s
    print(f"{t:<13}{s[0]:>+9.2f}{s[1]:>+9.2f}{s[2]:>+10.2f}  |{s[3]:>+9.2f}{s[4]:>+9.2f}{s[5]:>+10.2f}")
print("-"*72)
M = np.array([R[t] for t in R])
gen = np.array([R[t] for t in R if t.startswith("W")])
lab = ["lr HWHM", "lr curv", "lr integ", "lz HWHM", "lz curv", "lz integ"]
print(f"\nIC-to-IC SPREAD across the {len(gen)} generic seeds (sd of the slope):")
for i, L in enumerate(lab):
    v = gen[:, i]; v = v[np.isfinite(v)]
    print(f"   {L:<10} sd = {v.std():.3f}   mean = {v.mean():+.3f}   range [{v.min():+.2f},{v.max():+.2f}]")
lr_sd = [gen[:, i][np.isfinite(gen[:, i])].std() for i in (0, 1, 2)]
lz_sd = [gen[:, i][np.isfinite(gen[:, i])].std() for i in (3, 4, 5)]
print(f"\n   radial sd by estimator: {np.round(lr_sd,3)}   axial sd: {np.round(lz_sd,3)}")
print(f"   ratio radial/axial per estimator: {np.round(np.array(lr_sd)/np.maximum(lz_sd,1e-9),2)}")
ok = all(a > 2*b for a, b in zip(lr_sd, lz_sd))
print(f"\nVERDICT: {'SPREAD SURVIVES all three estimators -> physical' if ok else 'spread is estimator-dependent -> NOT established'}")
