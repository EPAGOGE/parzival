"""Close the XMAX axis the ej audit surfaced.  Converge the SAME profile on two more
outer-domain truncations and re-measure the two numbers the verdict rests on:
the spectral abscissa and eps* = min_(Re z >= 0) sigma_min(zI - L).

The far-field symbol says the untruncated essential spectrum is the imaginary axis,
so the prediction is that both margins SHRINK as XMAX grows.  If they do not, the
untruncated reading is wrong and the margin is a genuine spectral gap."""
import sys, os, time
import numpy as np
import scipy.linalg as sla
os.chdir("/private/tmp/claude-501/-Users-epagogellc/"
         "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
sys.path.insert(0, os.getcwd())
import q345
log = q345.log
pc = q345.spectrum.pc

CASES = [("X18", (0.0, 2.0, 11.0, 18.0)),
         ("X25", (0.0, 2.0, 15.0, 25.0)),
         ("X35", (0.0, 2.0, 21.0, 35.0))]
DEGS, NB, EPSB = (16, 40, 12), 36, 1e-4

for tag, edges in CASES:
    t0 = time.time()
    S, z, r, info = pc.converge(edges=edges, degs=DEGS, Nb=NB, eps_b=EPSB, tol=1e-11)
    if not info.get("converged"):
        log(f"[{tag}] XMAX={edges[-1]}  NOT CONVERGED: {info}  [{time.time()-t0:.0f}s]")
        continue
    log(f"[{tag}] XMAX={edges[-1]:5.1f}  converged=True  alpha={info['alpha']:+.8f}  "
        f"h_id={info['h_id']:+.4e}  ||F||={r:.3e}  passes={info['passes']}  "
        f"[{time.time()-t0:.0f}s]")
    real = q345.spectrum.Realization(S, z)
    M, _, _ = q345.dense_M_blocked(real)
    w = q345.verify_M(real, M, ntrial=2)
    Ld = M - real.Bc @ np.linalg.solve(real.CgBc, real.Cg @ M)
    del M
    V = q345.householder2(real.Cg.T)
    q345.congruence(Ld, V)
    lead = float(np.linalg.norm(Ld[:2, :]))
    L = np.ascontiguousarray(Ld[2:, 2:])
    del Ld
    nrm = float(np.linalg.norm(L, 2))
    om = float(sla.eigvalsh(0.5 * (L + L.T),
                            subset_by_index=[L.shape[0]-1, L.shape[0]-1])[0])
    ss = q345.SchurSigma(L)
    ss.tol = 1e-9
    lam = ss.ev
    log(f"        M-check {w:.2e}  Householder lead {lead:.2e}  n={ss.n}  "
        f"||L||={nrm:.6e}  omega={om:+.6e}  omega/||L||={om/nrm:.5f}")
    ys = np.unique(np.concatenate([np.arange(0.0, 12.0001, 0.05),
                                   np.arange(12.0, 130.001, 0.5),
                                   np.arange(130.0, nrm + 6.0, 5.0)]))
    sy = q345.scan(ss, [1j * y for y in ys], tag=tag, every=0)
    k = int(np.argmin(sy))
    yr = np.linspace(ys[max(k-1,0)], ys[min(k+1,ys.size-1)], 41)
    sr = q345.scan(ss, [1j*y for y in yr], tag=tag, every=0)
    eps = float(min(sy.min(), sr.min()))
    log(f"        SPECTRAL ABSCISSA = {lam.real.max():+.8e}   "
        f"(rightmost {lam[np.argmax(lam.real)].real:+.6f}"
        f"{lam[np.argmax(lam.real)].imag:+.6f}i)   count Re>0 = {int((lam.real>0).sum())}")
    log(f"        eps* = {eps:.8e}   at y = {yr[int(np.argmin(sr))]:+.4f}   "
        f"sigma_min(0) = {sy[0]:.6e}   eps*/||L|| = {eps/nrm:.4e}")
    log(f"        e^(a0 XMAX) = {np.exp(S.a0*edges[-1]):.4e}")
    del ss, L
log("DONE")
