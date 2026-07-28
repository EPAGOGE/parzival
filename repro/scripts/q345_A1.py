"""A1 -- Schur form of the production quotient, the production-scale route gate,
the spectrum with its conditioning, and the numerical abscissa."""
import sys, time
import numpy as np
import scipy.linalg as sla
sys.path.insert(0, "/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
import q345
log = q345.log

LAB = sys.argv[1] if len(sys.argv) > 1 else "A"
LP = f"q345_Lred_{LAB}.npy"
import pathlib, os
os.chdir("/private/tmp/claude-501/-Users-epagogellc/"
         "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")

if pathlib.Path(LP).exists():
    Lred = np.load(LP)
    log(f"[{LAB}] loaded Lred {Lred.shape}")
else:
    d = q345.build_quotient(LAB)
    Lred = d["Lred"]
    np.save(LP, Lred)

n = Lred.shape[0]

# ---- numerical abscissa BEFORE the Schur (needs Lred itself) ---------------
t0 = time.time()
Her = 0.5 * (Lred + Lred.T)
ev_h = sla.eigvalsh(Her, subset_by_index=[0, 0])[0], sla.eigvalsh(
    Her, subset_by_index=[n - 1, n - 1])[0]
del Her
log(f"[{LAB}] field of values, Re range = [{ev_h[0]:+.6e}, {ev_h[1]:+.6e}]"
    f"   omega(L) = {ev_h[1]:+.6e}   [{time.time()-t0:.0f}s]")

# ---- Schur ----------------------------------------------------------------
ss = q345.SchurSigma(Lred)
log(f"[{LAB}] complex Schur n={n}  [{ss.t_schur:.0f}s]")
np.save(f"q345_ev_{LAB}.npy", ss.ev)
np.save(f"q345_T_{LAB}.npy", ss.T)

lam = ss.ev
log(f"[{LAB}] spectrum: max Re = {lam.real.max():+.6e}   min Re = {lam.real.min():+.6e}"
    f"   max |Im| = {np.abs(lam.imag).max():.4e}")
rhp = np.argsort(-lam.real)[:20]
log(f"[{LAB}] 20 rightmost Schur-diagonal eigenvalues:")
for k in rhp:
    log(f"      {lam[k].real:+.8e} {lam[k].imag:+.8e}i")
npos = int((lam.real > 0).sum())
log(f"[{LAB}] count Re(lambda) > 0 : {npos}   of {n}")

# ---- production-scale route gate: dense Schur vs the sparse bordered LU ----
log(f"\n[{LAB}] PRODUCTION ROUTE GATE  (dense Schur sigma_min vs spectrum.py sparse LU)")
REF = {"A": {0.0: 2.518838e+02, 0.5: 1.1130303058e+02, 1.0: 7.83376e+01,
             2.0: 4.47738e+01, 1.0j: 2.5232276633e+02, 2.0j: 2.52088e+02},
       "B": {1.0j: 2.90407e+02, 2.0j: 2.41209e+02, 1.0: 1.20270e+02,
             2.0: 7.30242e+01},
       "C": {1.0j: 2.55024e+02, 2.0j: 2.54900e+02, 1.0: 7.84715e+01,
             2.0: 4.49965e+01}}
worst = 0.0
for z, ref in REF.get(LAB, {}).items():
    s, it = ss.sigma_min(complex(z), warm=False)
    rel = abs(1.0 / s - ref) / ref
    worst = max(worst, rel)
    log(f"      z={complex(z).real:+.2f}{complex(z).imag:+.2f}i   "
        f"||R|| dense={1/s:.10e}   sparse={ref:.10e}   rel={rel:.3e}  [{it} it]")
log(f"      worst = {worst:.3e}   (bar 1e-6, the sparse power iteration's own tol)"
    f"  -> {'PASS' if worst < 1e-6 else 'FAIL'}")
log("DONE")
