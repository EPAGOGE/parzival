"""A2 (v2) -- Q3 pseudospectra + THE CROSSING VERDICT, Q4 Kreiss, Q5 conditioning.

The crossing verdict is a ONE-DIMENSIONAL measurement, and the chain that makes it
so is worth stating because it is what replaces the covering argument this file
originally attempted and refused:

  (1) the Lyapunov/Cholesky certificate (q345_lyap.py) settles that L has no
      eigenvalue with Re >= 0 -- no eigenvalue is computed or trusted;
  (2) sigma_min(zI-L) >= |z| - ||L||, so nothing happens outside |z| <= ||L||;
  (3) ||R(z)|| has NO LOCAL MAXIMUM in the resolvent set (Davies-Shargorodsky), so
      with (1) its maximum over the closed right half plane is attained ON THE
      IMAGINARY AXIS.

Therefore  eps* = min_{Re z >= 0} sigma_min(zI - L) = min_y sigma_min(iy I - L),
a 1-D scan.  The 2-D maps below are the empirical control on (3), and the picture.
"""
import sys, time, os
import numpy as np
import scipy.linalg as sla
os.chdir("/private/tmp/claude-501/-Users-epagogellc/"
         "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
sys.path.insert(0, os.getcwd())
import q345
log = q345.log

LAB = sys.argv[1] if len(sys.argv) > 1 else "A"
T = np.load(f"q345_T_{LAB}.npy")
ss = q345.SchurSigma(None, T=T)
n = ss.n
lam = ss.ev
NORM_L = {"A": 1.112468e+03, "B": 2.453998e+03, "C": 1.103660e+03}[LAB]
OMEGA = {"A": 5.508533e+02, "B": 1.218808e+03, "C": 5.465627e+02}[LAB]
log(f"[{LAB}] n={n}  ||L||={NORM_L:.6e}  omega(L)={OMEGA:+.6e}  "
    f"Schur-diagonal spectral abscissa = {lam.real.max():+.8e}")
out = {}

# ---------------------------------------------------------------------------
# Q3a  the imaginary axis IS the verdict
# ---------------------------------------------------------------------------
log(f"\n[{LAB}] Q3a  IMAGINARY AXIS  (carries the RHP minimum; |y| <= ||L|| suffices)")
ss.tol = 1e-9
ys = np.unique(np.concatenate([np.arange(0.0, 12.0001, 0.05),
                               np.arange(12.0, 130.001, 0.5),
                               np.arange(130.0, NORM_L + 6.0, 5.0)]))
t0 = time.time()
sy = q345.scan(ss, [1j * y for y in ys], tag=f"{LAB} imag", every=200)
log(f"    {ys.size} points  [{time.time()-t0:.0f}s]")
k = int(np.argmin(sy))
log(f"    sigma_min(0)      = {sy[0]:.6e}     ||R(0)||   = {1/sy[0]:.6e}")
log(f"    MIN on the axis   = {sy[k]:.6e}     ||R||_max  = {1/sy[k]:.6e}   at y = {ys[k]:+.4f}")
log(f"    MAX on the axis   = {sy.max():.6e}  at y = {ys[np.argmax(sy)]:+.1f}")
kk = np.argsort(sy)[:10]
log("    10 lowest:  " + "  ".join(f"{ys[i]:.2f}i:{sy[i]:.4e}" for i in sorted(kk)))
# local refinement around the argmin
lo, hi = ys[max(k - 1, 0)], ys[min(k + 1, ys.size - 1)]
yr = np.linspace(lo, hi, 61)
sr = q345.scan(ss, [1j * y for y in yr], tag=f"{LAB} imagref", every=0)
kr = int(np.argmin(sr))
log(f"    refined over y in [{lo:.3f},{hi:.3f}]:  eps* = {sr[kr]:.8e} at y = {yr[kr]:+.5f}")
np.savez(f"q345_imag_{LAB}.npz", ys=ys, sy=sy, yr=yr, sr=sr)
eps_star = float(min(sr.min(), sy.min()))
out["eps_star"] = eps_star
out["eps_star_rel"] = eps_star / NORM_L
log(f"\n    ===> eps* = min_(Re z >= 0) sigma_min(zI-L) = {eps_star:.8e}")
log(f"         relative to ||L||:  {eps_star/NORM_L:.6e}")

# ---------------------------------------------------------------------------
# Q3b  the picture: SPEC window, and a coarse RHP map as the control on (3)
# ---------------------------------------------------------------------------
ss.tol = 1e-5
log(f"\n[{LAB}] Q3b  SPEC WINDOW  Re in [-2,2], Im in [0,2]  (real L: mirror for Im<0)")
xw = np.linspace(-2.0, 2.0, 81)
yw = np.linspace(0.0, 2.0, 41)
t0 = time.time()
Sw = q345.grid_map(ss, xw, yw, tag=f"{LAB} win")
np.savez(f"q345_win_{LAB}.npz", xs=xw, yg=yw, S=Sw)
jw, iw = np.unravel_index(np.argmin(Sw), Sw.shape)
log(f"    min = {Sw.min():.6e} at z = {xw[iw]:+.3f}{yw[jw]:+.3f}i  "
    f"(||R||max = {1/Sw.min():.4e})   [{time.time()-t0:.0f}s]")
log(f"    {'eps':>10s} {'rightmost Re of the eps-pseudospectrum':>42s}")
for eps in (1e0, 3e-1, 1e-1, 3e-2, 1e-2, 5e-3, eps_star * 1.001, eps_star * 0.999,
            1e-3, 1e-6, 1e-9):
    ins = Sw <= eps
    if ins.any():
        rmax = xw[np.argwhere(ins)[:, 1]].max()
        log(f"    {eps:10.4e} {rmax:>+20.4f}   "
            f"{'REACHES Re>0' if rmax > 0 else 'confined to Re<0'}")
    else:
        log(f"    {eps:10.4e} {'empty in this window':>42s}")

log(f"\n[{LAB}] Q3c  COARSE RHP MAP  (control: no interior local max of ||R||)")
h = 20.0
xg = np.arange(0.0, min(NORM_L, OMEGA) + h, h)
yg = np.arange(0.0, NORM_L + h, h)
t0 = time.time()
Sg = q345.grid_map(ss, xg, yg, tag=f"{LAB} rhp")
np.savez(f"q345_rhp_{LAB}.npz", xs=xg, yg=yg, S=Sg, h=h)
jg, ig = np.unravel_index(np.argmin(Sg), Sg.shape)
log(f"    {xg.size}x{yg.size} pts, h={h}   min = {Sg.min():.6e} at "
    f"z = {xg[ig]:+.0f}{yg[jg]:+.0f}i   [{time.time()-t0:.0f}s]")
col0 = Sg[:, 0]
interior_min = Sg[:, 1:].min() if Sg.shape[1] > 1 else np.inf
log(f"    min on the Re=0 column = {col0.min():.6e};  min over Re>0 = {interior_min:.6e}"
    f"   -> {'axis carries it' if col0.min() <= interior_min else 'INTERIOR MINIMUM (theorem violated)'}")

log(f"\n[{LAB}] Q3d  NEAR-AXIS BAND  Re in [0,16], Im in [0,60]")
xb = np.arange(0.0, 16.001, 0.5)
yb = np.arange(0.0, 60.001, 0.5)
t0 = time.time()
Sb = q345.grid_map(ss, xb, yb, tag=f"{LAB} band")
np.savez(f"q345_band_{LAB}.npz", xs=xb, yg=yb, S=Sb)
jb, ib = np.unravel_index(np.argmin(Sb), Sb.shape)
log(f"    min = {Sb.min():.6e} at z = {xb[ib]:+.2f}{yb[jb]:+.2f}i   [{time.time()-t0:.0f}s]")
log(f"    min on Re=0 column = {Sb[:,0].min():.6e};  min over Re>0 = {Sb[:,1:].min():.6e}")

# ---------------------------------------------------------------------------
# Q4  Kreiss constant
# ---------------------------------------------------------------------------
log(f"\n[{LAB}] Q4  KREISS CONSTANT   K = sup_(Re z>0) Re(z) ||R(z)||")
ss.tol = 1e-8
rr = np.concatenate([np.arange(0.05, 4.0, 0.05), np.arange(4.0, 40.0, 0.5),
                     np.arange(40.0, 620.0, 10.0)])
sr2 = q345.scan(ss, list(rr.astype(complex)), tag=f"{LAB} real", every=100)
kr = rr / sr2
m = int(np.argmax(kr))
Kgrid = 0.0
zK = None
for arr, X, Y in ((Sg, xg, yg), (Sb, xb, yb), (Sw, xw, yw)):
    for jj in range(arr.shape[0]):
        for ii in range(arr.shape[1]):
            if X[ii] > 0 and X[ii] / arr[jj, ii] > Kgrid:
                Kgrid = X[ii] / arr[jj, ii]
                zK = complex(X[ii], Y[jj])
K = max(kr[m], Kgrid)
log(f"    real-axis scan:  K >= {kr[m]:.6e} at z = {rr[m]:+.3f} "
    f"(sigma_min = {sr2[m]:.4e})")
log(f"    2-D grids:       K >= {Kgrid:.6e} at z = {zK}")
log(f"    ===> K >= {K:.6e}   =>   sup_t ||e^(tL)||_2 >= {K:.6e}")
log(f"    omega(L) = {OMEGA:+.6e} = d/dt ||e^(tL)|| at t=0  (initial growth rate)")
np.savez(f"q345_real_{LAB}.npz", rr=rr, sr=sr2)
out["K"] = float(K)

# ---------------------------------------------------------------------------
# Q5a  conditioning of the rightmost eigenvalues
# ---------------------------------------------------------------------------
log(f"\n[{LAB}] Q5a  EIGENVALUE CONDITIONING   kappa(lam) = lim_(r->0) r / min_th"
    f" sigma_min((lam + r e^(i th))I - L)")
ss.tol = 1e-10
order = np.argsort(-lam.real)
seen = []
cond = []
for idx in order:
    v = lam[idx]
    if any(abs(v - w) < 1e-8 for w in seen):
        continue
    seen.append(v)
    log(f"    lambda = {v.real:+.8e} {v.imag:+.8e}i")
    ks = []
    for r in (1e-1, 1e-2, 1e-3, 1e-4):
        s = min(ss.sigma_min(v + r * np.exp(1j * th))[0]
                for th in np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ks.append(r / s)
        log(f"        r={r:.0e}   min_th sigma_min = {s:.6e}   kappa ~ {r/s:.6e}")
    cond.append((v, ks[-1]))
    if len(seen) >= 4:
        break
kap = cond[0][1]
log(f"    rightmost eigenvalue {cond[0][0].real:+.8f}{cond[0][0].imag:+.8f}i  "
    f"kappa = {kap:.4e}")
log(f"    => a perturbation of size delta moves it by up to kappa*delta; to reach"
    f" Re = 0 needs delta >= {abs(cond[0][0].real)/kap:.6e}")
out["kappa_rightmost"] = float(kap)

log(f"\n[{LAB}] SUMMARY  " + "   ".join(f"{k}={v:.6e}" if isinstance(v, float)
                                        else f"{k}={v}" for k, v in out.items()))
log("DONE")
