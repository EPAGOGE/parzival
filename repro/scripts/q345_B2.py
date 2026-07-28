"""Q6 -- the decisive quantity at the second resolution: eps*, the Kreiss constant,
and the conditioning of the rightmost eigenvalue.  No 2-D maps: the crossing verdict
is a 1-D object once the Lyapunov certificate has settled the RHP spectrum."""
import sys, time, os
import numpy as np
os.chdir("/private/tmp/claude-501/-Users-epagogellc/"
         "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
sys.path.insert(0, os.getcwd())
import q345
log = q345.log
LAB = sys.argv[1]
NORM_L = {"A": 1.112468e+03, "B": 2.453998e+03, "C": 1.103660e+03}[LAB]
OMEGA = {"A": 5.508533e+02, "B": 1.218808e+03, "C": 5.465627e+02}[LAB]
ss = q345.SchurSigma(None, T=np.load(f"q345_T_{LAB}.npy"))
lam = ss.ev
log(f"[{LAB}] n={ss.n}  ||L||={NORM_L:.6e}  omega={OMEGA:+.6e}  "
    f"Schur abscissa={lam.real.max():+.8e}")

ss.tol = 1e-9
ys = np.unique(np.concatenate([np.arange(0.0, 12.0001, 0.05),
                               np.arange(12.0, 130.001, 0.5),
                               np.arange(130.0, NORM_L + 6.0, 5.0)]))
t0 = time.time()
sy = q345.scan(ss, [1j * y for y in ys], tag=f"{LAB} imag", every=300)
k = int(np.argmin(sy))
log(f"    {ys.size} pts [{time.time()-t0:.0f}s]  sigma_min(0) = {sy[0]:.6e}   "
    f"||R(0)|| = {1/sy[0]:.6e}")
log(f"    MIN on the axis = {sy[k]:.6e} at y = {ys[k]:+.4f}   ||R||max = {1/sy[k]:.6e}")
yr = np.linspace(ys[max(k-1,0)], ys[min(k+1,ys.size-1)], 61)
sr = q345.scan(ss, [1j*y for y in yr], tag="ref", every=0)
eps = float(min(sr.min(), sy.min()))
log(f"    refined: eps* = {eps:.8e} at y = {yr[int(np.argmin(sr))]:+.5f}")
log(f"    ===> eps* = {eps:.8e}   eps*/||L|| = {eps/NORM_L:.6e}")
np.savez(f"q345_imag_{LAB}.npz", ys=ys, sy=sy, yr=yr, sr=sr)

ss.tol = 1e-8
rr = np.concatenate([np.arange(0.05, 4.0, 0.05), np.arange(4.0, 40.0, 0.5),
                     np.arange(40.0, OMEGA + 40.0, 10.0)])
sr2 = q345.scan(ss, list(rr.astype(complex)), tag=f"{LAB} real", every=150)
kr = rr / sr2
m = int(np.argmax(kr))
log(f"    KREISS: K >= {kr[m]:.6e} at z = {rr[m]:+.3f}  (sigma_min = {sr2[m]:.4e})")
np.savez(f"q345_real_{LAB}.npz", rr=rr, sr=sr2)

ss.tol = 1e-10
log(f"[{LAB}] conditioning of the rightmost eigenvalues:")
seen = []
for idx in np.argsort(-lam.real):
    v = lam[idx]
    if any(abs(v - w) < 1e-8 for w in seen):
        continue
    seen.append(v)
    log(f"    lambda = {v.real:+.8e}{v.imag:+.8e}i")
    for r in (1e-2, 1e-3, 1e-4):
        s = min(ss.sigma_min(v + r*np.exp(1j*th))[0]
                for th in np.linspace(0, 2*np.pi, 8, endpoint=False))
        log(f"        r={r:.0e}  min_th sigma_min={s:.6e}   kappa ~ {r/s:.6e}")
    if len(seen) >= 3:
        break
log("DONE")
