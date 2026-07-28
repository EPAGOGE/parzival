"""Q2: the exactly-known nulls, the quotient, and the gate.

Two candidate null directions, both derived (not guessed):

 (1) GRADING  v_g = (A, 2B, P, cl, cw).  Exact ALGEBRAIC symmetry (pointwise
     multiplication) -> exact to roundoff on covariant rows.  Already measured.

 (2) DILATION v_d.  The continuum problem on the wedge is invariant under
     y -> lambda y with Om(y)->Om(lam y), Th->lam^-1 Th(lam y), Psi->lam^-2 Psi(lam y),
     cl,cw FIXED.  Generator: delta = d_s (= g d_xi) plus the weight terms.  In the
     divided corner-frame variables (Om = e^{a0 xi} xi A, Bf = e^{(1+2a0)xi} xi^2 B,
     Psi = e^{mu xi} xi^2 P, d_s = g d_xi, g = xi G1):

         dA = G1 * (LA A)
         dB = G1 * (LB2 B) - B
         dP = G1 * (LPmu P) - 2 P
         dcl = dcw = 0

     This is a DIFFERENTIAL symmetry, so the DISCRETE residual is only covariant to
     truncation error -- the measurement below decides whether it may be deflated.

Corner action (the reason both matter): v_g moves (wx, thxx) by (1,2) in relative
units, v_d by (1,1).  Neither, nor any combination, fixes both -- so the pinned root
IS isolated.  But in the CONTINUUM the two gauge rows impose the same two numbers the
corner pins already impose, so the continuum system is consistent-but-overdetermined
by 2 and the discrete Jacobian should be rank-deficient by 2 up to truncation error.
PREDICTION UNDER TEST: J has TWO singular values at the ~2.5e-6 scale (not one), and
the corresponding right-singular subspace is close to span{v_g, v_d}.
"""
import importlib.util, pathlib, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)

FIELD = SCR / "hunt_fields/rung_00_a-0.344712.npz"
d = np.load(FIELD)
a, z = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36,
                       eps_b=1e-4, alpha=a)
S.adopt_seed(z)
n2 = S.Nx * S.Nb
N = z.size
F = S.residual(z)
J = S.jacobian(z).tocsr()
print(f"root: alpha={a:+.8f}  ||F||_rms={np.linalg.norm(F)/np.sqrt(F.size):.3e}  "
      f"h_id={S.h_id(z):+.3e}  n={N}  (Nx={S.Nx}, Nb={S.Nb}, n2={n2})", flush=True)

A, B, P, cl, cw = S.unpack(z)
x, G1, mu, a0 = S.XI, S.G1, S.mu, S.a0

# ---------------------------------------------------------------- generators --
v_g = np.concatenate([z[:n2], 2.0 * z[n2:2 * n2], z[2 * n2:3 * n2], [z[-2]], [z[-1]]])
LAa = A + x * ((S.Dx @ A) + a0 * A)
LB2b = 2.0 * B + x * ((S.Dx @ B) + (1.0 + 2.0 * a0) * B)
LPp = 2.0 * P + x * ((S.Dx @ P) + mu * P)
dA = G1 * LAa
dB = G1 * LB2b - B
dP = G1 * LPp - 2.0 * P
v_d = np.concatenate([dA.ravel(), dB.ravel(), dP.ravel(), [0.0, 0.0]])
v_g /= np.linalg.norm(v_g); v_d /= np.linalg.norm(v_d)
print(f"    |cos(v_g, v_d)| = {abs(v_g @ v_d):.6f}", flush=True)

# ------------------------------------------------------------- row bookkeeping --
pin = np.concatenate([S.rT_pin, S.rT_pin + n2])
c0 = np.concatenate([S.rT_c0, S.rT_c0 + n2])
prow = np.arange(2 * n2, 3 * n2)
grow = np.array([N - 2, N - 1])
alg = np.concatenate([pin, c0, prow, grow])
dyn = np.setdiff1d(np.arange(N), alg)
mask = np.zeros(N); mask[dyn] = 1.0
E = sp.diags(mask, format="csr")
print(f"    pencil (E,J): rank(E)={len(dyn)} of {N}   E-nullity={N-len(dyn)} "
      f"= P {len(prow)} + c 2 + pins {len(pin)} + C0 {len(c0)}", flush=True)

Jn = np.sqrt((J.multiply(J)).sum())
print("\n[A] EULER / COVARIANCE TEST  (raw ||J.v||, and where it sits)", flush=True)
for nm, v in (("grading v_g", v_g), ("dilation v_d", v_d)):
    w = J @ v
    tot = np.linalg.norm(w)
    cov = np.linalg.norm(w[dyn])
    print(f"    {nm:<12s} ||J.v||={tot:.4e}  ||J.v||/||J||={tot/Jn:.3e}   "
          f"COVARIANT-row part={cov:.4e} ({cov/tot:.4%})   "
          f"pins+gauge+C0+P={np.linalg.norm(w[alg]):.4e}", flush=True)
print("    (v_g covariant part ~1e-12 = exact algebraic symmetry; v_d covariant part"
      "\n     is the DISCRETE truncation error of a differential symmetry.)", flush=True)

# corner action: relative motion of the two pinned amplitudes
def corner_amp(vv):
    dAf = vv[:n2].reshape(S.Nx, S.Nb); dBf = vv[n2:2 * n2].reshape(S.Nx, S.Nb)
    # A(0,b) = wx cos b ; B(0,b) = thxx/2 cos^2 b  -> read the amplitudes
    cb = np.cos(S.b)
    wx = float(dAf[0] @ cb / (cb @ cb)); th = 2.0 * float(dBf[0] @ cb**2 / (cb**2 @ cb**2))
    return wx / S.wx, th / S.thxx
for nm, v in (("grading v_g", v_g), ("dilation v_d", v_d)):
    rw, rt = corner_amp(v * np.linalg.norm(v_g))
    print(f"    {nm:<12s} corner action (d wx/wx, d thxx/thxx) ~ "
          f"({rw:+.4e}, {rt:+.4e})   ratio {rt/rw if rw else np.nan:+.4f}", flush=True)

# ------------------------------------------------- smallest singular subspace --
print("\n[B] SMALLEST SINGULAR SUBSPACE of J  (block inverse subspace iteration)",
      flush=True)
lu = spla.splu(J.tocsc()); luT = spla.splu(J.T.tocsc())
k = 6
rng = np.random.default_rng(11)
X = rng.standard_normal((N, k))
X, _ = np.linalg.qr(X)
for it in range(80):
    Y = np.column_stack([lu.solve(luT.solve(X[:, j])) for j in range(k)])
    X, _ = np.linalg.qr(Y)
JX = J @ X
U_, s_, Vt_ = la.svd(JX, full_matrices=False)
order = np.argsort(s_)
sig = s_[order]
Xr = X @ Vt_.T[:, order]                 # Ritz right-singular vectors
print("    smallest Ritz singular values: " + " ".join(f"{v:.4e}" for v in sig),
      flush=True)
Vsym = np.column_stack([v_g, v_d])
Qs, _ = np.linalg.qr(Vsym)
for i in range(k):
    ov = np.linalg.norm(Qs.T @ Xr[:, i])
    print(f"      sigma_{i+1} = {sig[i]:.4e}   |proj onto span(v_g,v_d)| = {ov:.4f}"
          f"   |<.,v_g>|={abs(Xr[:,i]@v_g):.4f}  |<.,v_d>|={abs(Xr[:,i]@v_d):.4f}",
          flush=True)
sub2 = Xr[:, :2]
ang = la.subspace_angles(sub2, Vsym)
print(f"    principal angles between the 2 smallest right-sing. vecs and "
      f"span(v_g,v_d): {np.degrees(ang)} deg", flush=True)
# where do the LEFT singular vectors live?
for i in range(2):
    u = J @ Xr[:, i]; u /= np.linalg.norm(u)
    print(f"      left vec {i+1}: on gauge rows {np.linalg.norm(u[grow]):.4f}, "
          f"corner+axis pins {np.linalg.norm(u[pin]):.4f}, "
          f"C0 {np.linalg.norm(u[c0]):.4f}, P {np.linalg.norm(u[prow]):.4f}, "
          f"covariant {np.linalg.norm(u[dyn]):.4f}", flush=True)

# ------------------------------------------------------- G1: deflated sigma_min --
def sigma_min_deflated(defl, iters=120, seed=3):
    Q = np.linalg.qr(np.column_stack(defl))[0] if defl else None
    xv = np.random.default_rng(seed).standard_normal(N)
    if Q is not None:
        xv -= Q @ (Q.T @ xv)
    xv /= np.linalg.norm(xv)
    for _ in range(iters):
        xv = lu.solve(luT.solve(xv))
        if Q is not None:
            xv -= Q @ (Q.T @ xv)
        xv /= np.linalg.norm(xv)
    return float(np.linalg.norm(J @ xv)), xv

print("\n[C] GATE G1 -- sigma_min on the deflated complement", flush=True)
s0, x0 = sigma_min_deflated([])
s1, x1 = sigma_min_deflated([v_g])
s2, x2 = sigma_min_deflated([v_g, v_d])
print(f"    sigma_min(J)                      = {s0:.4e}   "
      f"|cos(.,v_g)|={abs(x0@v_g):.4f}", flush=True)
print(f"    sigma_min | v_g deflated          = {s1:.4e}   "
      f"({s1/s0:.3g}x)   |cos(.,v_d)|={abs(x1@v_d):.4f}", flush=True)
print(f"    sigma_min | {{v_g,v_d}} deflated    = {s2:.4e}   ({s2/s0:.3g}x)",
      flush=True)
print(f"    ||Pi v_g||/||v_g|| with Pi = I - QQ^T over {{v_g,v_d}}: "
      f"{np.linalg.norm(v_g - Qs @ (Qs.T @ v_g)):.3e}", flush=True)

# ----------------------------------------- how much of v_g the pins actually see --
print(f"\n[D] ADMISSIBILITY of the symmetry directions (state space = free dofs)",
      flush=True)
for nm, v in (("v_g", v_g), ("v_d", v_d)):
    fld = np.concatenate([v[:n2], v[n2:2 * n2]])
    ppart = np.linalg.norm(v[pin]); cpart = np.linalg.norm(v[c0])
    print(f"    {nm}: ||v on pinned nodes||/||v_field|| = "
          f"{ppart/np.linalg.norm(fld):.4f}   on C0 nodes {cpart/np.linalg.norm(fld):.4f}"
          f"   -> {'INADMISSIBLE' if ppart/np.linalg.norm(fld) > 1e-8 else 'admissible'}",
          flush=True)

# --------------------------------------------- the realization: index-2 check --
print("\n[E] REALIZATION -- Hessenberg index-2 data", flush=True)
Bcol = J[:, [N - 2, N - 1]].toarray()
Cg = J[[N - 2, N - 1], :].toarray()
GN = Cg[:, :2 * n2] @ Bcol[:2 * n2, :]
print(f"    G N (2x2) =\n{GN}", flush=True)
print(f"    cond(G N) = {np.linalg.cond(GN):.4e}  det = {np.linalg.det(GN):+.4e}  "
      f"-> index {'2 (well posed)' if np.linalg.cond(GN) < 1e12 else 'DEGENERATE'}",
      flush=True)
print(f"    ||B on algebraic rows|| = {np.linalg.norm(Bcol[alg]):.3e} "
      f"(must be 0: c cannot force a constraint row)", flush=True)
print(f"    Cg support: A block {np.linalg.norm(Cg[:, :n2]):.4e}  "
      f"B block {np.linalg.norm(Cg[:, n2:2*n2]):.4e}  "
      f"P block {np.linalg.norm(Cg[:, 2*n2:3*n2]):.4e}  "
      f"c block {np.linalg.norm(Cg[:, 3*n2:]):.4e}", flush=True)

# ------------------------------ resolvent at z=0 on the quotient => sigma_min(L) --
print("\n[F] THE QUOTIENT THAT MATTERS: sigma_min of the EVOLUTION generator L",
      flush=True)
QG = np.linalg.qr(Cg[:, :].T)[0]         # 2 columns, in full z-space
QGd = QG[dyn, :]
QGd, _ = np.linalg.qr(QGd)               # orthonormal in the dynamic coordinates

def Rz_apply(f, luK, trans="N"):
    """f lives on the dynamic coords; returns (zI-L)^-1 f on the dynamic coords,
    projected onto ker(G).  Sparse bordered solve -- the pins/C0/P/gauge rows of
    (zE - J) enforce the whole constraint set."""
    f = f - QGd @ (QGd.T @ f)
    rhs = np.zeros(N); rhs[dyn] = f
    w = luK.solve(rhs, trans=trans)
    out = w[dyn]
    return out - QGd @ (QGd.T @ out)

def resolvent_norm(zval, iters=60, seed=5):
    K = (zval * E - J).tocsc()
    luK = spla.splu(K)
    v = np.random.default_rng(seed).standard_normal(len(dyn))
    v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(iters):
        w = Rz_apply(v, luK, "N")
        u = Rz_apply(w, luK, "T")
        nrm = np.linalg.norm(u)
        v = u / nrm
        s = nrm
    return np.sqrt(s), luK

R0, luK0 = resolvent_norm(0.0)
print(f"    ||R(0)|| = ||L^-1|| = {R0:.6e}   =>  sigma_min(L) = {1.0/R0:.6e}",
      flush=True)
print(f"    compare sigma_min(J) = {s0:.4e}  -> the quotient RAISES the smallest "
      f"scale by {(1.0/R0)/s0:.4g}x", flush=True)
# constraint sanity: output really in ker(G)?
ftest = np.random.default_rng(7).standard_normal(len(dyn))
wtest = Rz_apply(ftest, luK0, "N")
full = np.zeros(N); full[dyn] = wtest
print(f"    constraint check ||Cg w||/||w|| = "
      f"{np.linalg.norm(Cg @ full)/np.linalg.norm(full):.3e}", flush=True)
print("done", flush=True)
