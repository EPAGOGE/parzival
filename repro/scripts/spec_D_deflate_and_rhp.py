"""ADJUDICATION D -- (1) does deflating the grading change anything measurable?
                     (2) is there anything in the right half plane?

(1) THE G1 CLAUSE-2 DISPUTE.  D1 reports gate G1 PASSED by building an explicit
    projector over {v_g, v_d} (||Pi v_g||/||v_g|| = 3.9e-16) and quoting the 1566x rise
    of sigma_min.  D2 reports clause 2 FAILED for the STRUCTURAL projector (0.99998) and
    argues the explicit deflation is unjustified (leakage 0.62 ||L||, buys 7.6e-5).
    Both agree on every number; they disagree on which projector is the quotient.
    DECIDE IT ON AN OBSERVABLE, not on the gate wording: does the deflation move
    ||R(z)|| anywhere?  If it does not, the DAE restriction IS the quotient and the
    explicit deflation is cosmetic.  If it does, D1's projector is doing real work.

(2) THE RHP.  omega(L) = +550.85 in the raw norm, so the abscissa certificate is
    unavailable and the half plane must actually be searched.  Shift-invert Arnoldi on
    the pencil: OP = (sigma E - J)^-1 E has eigenvalues mu = 1/(sigma - lam), so it
    converges to the pencil eigenvalues NEAREST sigma and sends the 2912 infinite ones
    to mu = 0.  Every returned lam carries its own conditioning estimate
    sigma_min(lam E - J)/||lam E - J||, per the premortem rule.
"""
import importlib.util, pathlib, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)

d = np.load(SCR / "hunt_fields/rung_00_a-0.344712.npz")
a, zroot = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36,
                       eps_b=1e-4, alpha=a)
S.adopt_seed(zroot)
Nx, Nb = S.Nx, S.Nb; n2 = Nx * Nb
J = S.jacobian(zroot).tocsr(); N = J.shape[0]
liveT = np.setdiff1d(np.arange(n2), np.union1d(S.rT_pin, S.rT_c0))
fr = np.concatenate([liveT, n2 + liveT]); nf = fr.size
mask = np.zeros(N); mask[fr] = 1.0
E = sp.diags(mask, format="csc"); Jc = J.tocsc()
Cg = np.asarray(J[[N - 2, N - 1], :].todense())[:, fr]
Qc, _ = np.linalg.qr(Cg.T)
q = np.load(SCR / "quotient_state.npz")
w1 = q["w1"]                       # the grading shadow, normalize(P_kerCg v_g)
w1 = w1 / np.linalg.norm(w1)
print(f"grid ({Nx}x{Nb}) N={N} n_f={nf}   ||F||_rms="
      f"{np.linalg.norm(S.residual(zroot))/np.sqrt(N):.3e}  h_id={S.h_id(zroot):+.3e}")
print(f"grading shadow w1: ||Cg w1|| = {np.linalg.norm(Cg @ w1):.3e}  (it IS in ker Cg)")


def Rnorm(zz, extra=None, iters=80, seed=0, tol=1e-9):
    """||R(z)|| on ker(Cg), optionally with 'extra' orthonormal columns also deflated."""
    Q = Qc if extra is None else np.linalg.qr(np.column_stack([Qc, extra]))[0]
    def Pk(v): return v - Q @ (Q.conj().T @ v)
    Mz = (zz * E - Jc).tocsc()
    lu = spla.splu(Mz)
    def R(f, H_=False):
        rhs = np.zeros(N, dtype=complex); rhs[fr] = f
        return lu.solve(rhs, trans=("H" if H_ else "N"))[fr]
    v = Pk(np.random.default_rng(seed).standard_normal(nf).astype(complex))
    v /= np.linalg.norm(v); s = 0.0
    for it in range(iters):
        y = Pk(R(v)); w = Pk(R(y, True)); nn = np.linalg.norm(w)
        s_new = np.sqrt(nn)
        if it > 4 and abs(s_new - s) < tol * s_new:
            s = s_new; break
        v = w / nn; s = s_new
    return s, it + 1


print("\n=== (1) DOES DEFLATING THE GRADING MOVE ||R(z)|| ? ===")
print(f"{'z':>14s} {'||R|| structural':>20s} {'||R|| + grading defl':>22s} {'rel change':>13s}")
for zz in (0.0 + 0j, 0.5 + 0j, 1.0 + 0j, 0.0 + 1.0j, -0.5 + 0j):
    s0, i0 = Rnorm(zz)
    s1, i1 = Rnorm(zz, extra=w1[:, None])
    print(f"{zz.real:+7.2f}{zz.imag:+6.2f}i {s0:>20.10e} {s1:>22.10e} "
          f"{abs(s1-s0)/s0:>13.3e}   [{i0}/{i1} it]")

print("\n=== (2) SHIFT-INVERT ARNOLDI: pencil eigenvalues nearest RHP shifts ===")


def near(sigma, k=8, tol=1e-10):
    Mz = (sigma * E - Jc).tocsc()
    t0 = time.time(); lu = spla.splu(Mz); tlu = time.time() - t0
    OP = spla.LinearOperator((N, N), matvec=lambda v: lu.solve(E @ v),
                             dtype=complex if np.iscomplexobj(sigma) else float)
    mu = spla.eigs(OP, k=k, which="LM", return_eigenvectors=False, tol=tol,
                   maxiter=5000)
    mu = mu[np.abs(mu) > 1e-12]
    lam = sigma - 1.0 / mu
    return lam[np.argsort(-lam.real)], tlu


def cond_of(lam):
    """sigma_min(lam E - J) / ||lam E - J||_1  -- the premortem's required tag."""
    Kz = (lam * E - Jc).tocsc()
    nrm = spla.norm(Kz, 1)
    try:
        lu = spla.splu(Kz)
    except Exception:
        return 0.0, nrm
    v = np.random.default_rng(1).standard_normal(N).astype(complex); v /= np.linalg.norm(v)
    s = 0.0
    for _ in range(30):
        y = lu.solve(v); w = lu.solve(y, trans="H"); nn = np.linalg.norm(w)
        v = w / nn; s = 1.0 / np.sqrt(nn)
    return s, nrm


best = []
for sigma in (0.25, 1.0, 4.0, 20.0, 100.0, 400.0):
    lam, tlu = near(sigma, k=10)
    npos = int((lam.real > 0).sum())
    print(f"\n  sigma = {sigma:7.2f}   [LU {tlu:.1f}s]   {npos} of {len(lam)} have Re>0")
    for L_ in lam[:4]:
        print(f"      lam = {L_.real:+.6e} {L_.imag:+.6e}i    |lam-sigma| = "
              f"{abs(L_-sigma):.4e}")
    best.extend(list(lam[:3]))

best = np.array(best)
rhp = best[best.real > 0]
print(f"\n  ALL shifts pooled: {len(rhp)} candidates with Re>0 out of {len(best)}")
if len(rhp):
    order = np.argsort(-rhp.real)
    print("  conditioning of the RHP candidates (premortem rule: no bare eigenvalues):")
    for L_ in rhp[order][:6]:
        s, nrm = cond_of(L_)
        print(f"      lam = {L_.real:+.6e}{L_.imag:+.6e}i   sigma_min = {s:.4e}   "
              f"||lamE-J||_1 = {nrm:.4e}   relative = {s/nrm:.3e}")
else:
    print("  -> no right-half-plane candidate returned by any shift.")
