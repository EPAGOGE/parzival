"""THE RHP VERDICT, without trusting an eigenvalue and without covering the plane.

L is Hurwitz  <=>  the Lyapunov equation  L^T P + P L = -I  has a solution P > 0.
Bartels-Stewart returns P; CHOLESKY either succeeds or it does not.  That is an
inertia certificate: it never forms an eigenvalue, never needs a grid, and its
failure mode is a hard exception rather than a number to be squinted at.

It also pays for Q4.  With L^T P + P L = -I and P > 0,

    d/dt (x^T P x) = -||x||^2 <= -(1/lambda_max(P)) x^T P x
    ||e^(tL)||_P  <=  exp(-t / (2 lambda_max P))
    ||e^(tL)||_2  <=  sqrt(kappa(P)) exp(-t / (2 lambda_max P))
    sup_t ||e^(tL)||_2  <=  sqrt(kappa(P))

so the same object that decides stability also gives the UPPER bound on transient
growth that pairs with the Kreiss LOWER bound -- and the weight P is exactly the
'Chen-Hou-style non-diagonal weighted energy norm' open tension #18 asked for:
omega_P(L) <= -1/(2 lambda_max P) < 0 by construction, where every DIAGONAL weight
tried in that sweep left omega positive.
"""
import sys, os, time
import numpy as np
import scipy.linalg as sla
os.chdir("/private/tmp/claude-501/-Users-epagogellc/"
         "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
sys.path.insert(0, os.getcwd())
import q345
log = q345.log

LAB = sys.argv[1] if len(sys.argv) > 1 else "A"
L = np.load(f"q345_Lred_{LAB}.npy")
n = L.shape[0]
nrmL = {"A": 1.112468e+03, "B": 2.453998e+03, "C": 1.103660e+03}[LAB]
log(f"[{LAB}] Lyapunov inertia certificate   n={n}  ||L||={nrmL:.6e}")

t0 = time.time()
P = sla.solve_continuous_lyapunov(L.T, -np.eye(n))
t_lyap = time.time() - t0
P = 0.5 * (P + P.T)
res = L.T @ P + P @ L + np.eye(n)
rres = np.linalg.norm(res, 2) / max(np.linalg.norm(P, 2) * nrmL, 1.0)
log(f"[{LAB}] solved  [{t_lyap:.0f}s]   ||L^T P + P L + I||_2 / (||P|| ||L||) = {rres:.3e}")

try:
    C = sla.cholesky(P, lower=True)
    ok = True
except sla.LinAlgError as e:
    ok = False
    log(f"[{LAB}] CHOLESKY FAILED: {e}")
log(f"[{LAB}] Cholesky(P) -> {'SUCCEEDS' if ok else 'FAILS'}   "
    f"=>  L is {'HURWITZ (no eigenvalue with Re >= 0)' if ok else 'NOT Hurwitz'}")

t0 = time.time()
w = sla.eigvalsh(P)
log(f"[{LAB}] lambda_min(P) = {w[0]:.8e}   lambda_max(P) = {w[-1]:.8e}"
    f"   kappa(P) = {w[-1]/w[0]:.6e}   [{time.time()-t0:.0f}s]")
log(f"[{LAB}] decay rate      1/(2 lambda_max P)      = {1.0/(2*w[-1]):.8e}"
    f"   (compare spectral abscissa)")
log(f"[{LAB}] TRANSIENT GROWTH UPPER BOUND  sup_t ||e^(tL)||_2 <= sqrt(kappa P)"
    f" = {np.sqrt(w[-1]/w[0]):.6e}")
log(f"[{LAB}] omega_P(L) <= -1/(2 lambda_max P) = {-1.0/(2*w[-1]):.6e}"
    f"   <-- the P-norm makes the generator dissipative; every DIAGONAL weight"
    f" tried left omega > 0")
np.save(f"q345_Pdiag_{LAB}.npy", w)
log("DONE")
