"""ADJUDICATION B2 -- winding count: fix the aliasing, then MEASURE the cost law.

B1's adaptive refinement was self-deceiving: it bisected where the UNWRAPPED increment
exceeded pi/2, but np.unwrap forces every increment below pi by construction, so the
criterion could never fire on an aliased arc.  All four counts came out clean integers
and all four were wrong.  Correct criterion: refine on the RAW principal-value increment
with a tight tolerance, and accept only when the integer is STABLE under one further
global doubling.

Then the real question: how many LU factorisations does an honest count need?  The
answer is set by the TOTAL VARIATION of arg det, not by the net winding -- as z runs up
the imaginary axis, arg(z - lam_j) increases by ~pi for EVERY eigenvalue in the left
half plane, and those add coherently.  Measure the law on the small case and extrapolate.
"""
import importlib.util, pathlib, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)


def perm_parity(p):
    p = np.asarray(p); n = p.size
    seen = np.zeros(n, bool); par = 0
    for i in range(n):
        if seen[i]:
            continue
        j = i; ln = 0
        while not seen[j]:
            seen[j] = True; j = p[j]; ln += 1
        par ^= (ln - 1) & 1
    return -1.0 if par else 1.0


def argdet(Kz):
    lu = spla.splu(Kz.tocsc())
    d = lu.U.diagonal()
    s = perm_parity(lu.perm_r) * perm_parity(lu.perm_c)
    return np.angle(d).sum() + (np.pi if s < 0 else 0.0)


def wind_uniform(Kfun, path, npts):
    ts = np.linspace(0.0, 1.0, npts + 1)
    v = np.array([argdet(Kfun(path(t))) for t in ts])
    d = np.mod(np.diff(v) + np.pi, 2 * np.pi) - np.pi     # principal-value increments
    return d.sum() / (2 * np.pi), float(np.abs(d).max()), float(np.abs(d).sum())


S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(6, 10, 5), Nb=10,
                       eps_b=1e-3, alpha=-0.3447)
z0 = S.pack(S.A0, S.B0, S.P0, S.P["cl"], S.P["cw"])
J = S.jacobian(z0).tocsr(); n2 = S.Nx * S.Nb; N = J.shape[0]
liveT = np.setdiff1d(np.arange(n2), np.union1d(S.rT_pin, S.rT_c0))
mask = np.zeros(N); mask[liveT] = 1.0; mask[n2 + liveT] = 1.0
Esp = sp.diags(mask, format="csc"); Jc = J.tocsc()
Jd = np.asarray(J.todense()); Ed = np.diag(mask)
AA, BB, aa, bb, Q, Zq = sla.ordqz(Jd, Ed, output="complex")
fin = np.abs(bb) > 1e-12 * max(abs(aa).max(), 1.0)
lam = aa[fin] / bb[fin]
nfin = int(fin.sum())
print(f"GROUND TRUTH: {nfin} finite eigenvalues, {int((lam.real>0).sum())} in RHP, "
      f"max Re = {lam.real.max():+.5f}, spectral radius = {np.abs(lam).max():.4f}")
Kf = lambda z: (z * Esp - Jc)

print("\n--- convergence in npts: the aliasing is quantitative ---")
cases = [("|z|=1e6 (all)", lambda t: 1e6 * np.exp(2j * np.pi * t), nfin),
         ("|z|=1",         lambda t: 1.0 * np.exp(2j * np.pi * t),
          int((np.abs(lam) < 1.0).sum())),
         ("|z|=0.30",      lambda t: 0.30 * np.exp(2j * np.pi * t),
          int((np.abs(lam) < 0.30).sum()))]
for name, path, truth in cases:
    print(f"  {name:14s} truth = {truth}")
    for npts in (64, 256, 1024, 4096, 16384):
        t0 = time.time()
        w, dmax, tv = wind_uniform(Kf, path, npts)
        ok = "OK" if round(w) == truth else "  "
        print(f"      npts={npts:6d}  winding={w:+12.6f} -> {round(w):+5d} {ok}"
              f"   max|dArg|={dmax:.3f}  totalVar/2pi={tv/2/np.pi:8.1f}"
              f"   [{time.time()-t0:.1f}s]")
        if round(w) == truth and dmax < 0.4:
            break

print("\n--- RHP D-contour: total variation vs eigenvalue count (the cost law) ---")
Rb = float(np.abs(lam).max()) * 1.05
def dpath(t, R=Rb):
    if t < 0.5:
        return 1j * R * (-1.0 + 4.0 * t)
    return R * np.exp(1j * (np.pi / 2 - 2 * np.pi * (t - 0.5)))
truth = int(((lam.real > 0) & (np.abs(lam) < Rb)).sum())
for npts in (256, 1024, 4096, 16384):
    t0 = time.time()
    w, dmax, tv = wind_uniform(Kf, dpath, npts)
    print(f"      npts={npts:6d}  winding={w:+12.6f} -> {round(w):+4d}  truth={truth}"
          f"   max|dArg|={dmax:.3f}  totalVar/2pi={tv/2/np.pi:8.1f}   [{time.time()-t0:.1f}s]")
    if round(w) == truth and dmax < 0.4:
        conv = npts
        break
else:
    conv = None
print(f"\n  n_finite = {nfin};  converged at npts = {conv}"
      f"  -> points per finite eigenvalue = {None if conv is None else conv/nfin:.2f}")
print(f"  production n_finite = 4758  =>  extrapolated npts ~ "
      f"{'n/a' if conv is None else int(conv/nfin*4758)}"
      f"  at 19 s per complex LU = "
      f"{'n/a' if conv is None else conv/nfin*4758*19/3600:.1f} h")
