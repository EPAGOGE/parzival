"""WHAT ARE THE FAR-FIELD MODES OF THE POISSON OPERATOR, AND IS THE IMPOSED CONDITION
ADMITTING A GROWING ONE?

At g -> 1 the Poisson equation is

    Pt_xixi + 2 mu Pt_xi + mu^2 Pt + Pt_bb = -Ot ,    mu = 2 + a0 = 1.658

Separating Pt ~ e^{lam xi} phi_j(b) with -Db2 phi_j = (2j)^2 phi_j (Dirichlet on
b in [0, pi/2], so the eigenvalues should be the EVEN integers squared -- which is also
what the corner-Mellin conormal symbol predicts independently: alpha_j = j*pi/omega with
omega = pi/2):

    (lam + mu)^2 = (2j)^2   =>   lam = -mu +- 2j

so for EVERY j >= 1 the '+' branch has lam = 2j - 1.658 > 0 and GROWS. The physical
solution is Pt -> Pt_inf(b) (the particular solution driven by Ot -> Ot_inf) plus only the
DECAYING branch. polar_corner._build_poisson imposes d_xi Pt = 0 at xi = XMAX, which is
not that condition: it selects whatever admixture of the growing mode makes the derivative
vanish. This gate checks the arithmetic before any of it is rebuilt.
"""
import sys, pathlib, numpy as np, scipy.linalg as la
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
import importlib.util


def mod(name, fn):
    sp = importlib.util.spec_from_file_location(name, str(H / fn))
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m


pst = mod("pst", "polar_stability.py")
pc = mod("pc", "polar_corner.py")

N = 36
C = pc.Corner(N, N, 25.0)
mu, a0 = C.mu, C.a0
print(f"N={N}  a0={a0:.8f}  mu = 2+a0 = {mu:.8f}   g(last) = {C.g[-1]:.6e}")

D2i = C.Db2[1:-1, 1:-1]
ev = np.sort(la.eigvals(-D2i).real)[:8]
print("\n  -Db2 (Dirichlet interior) eigenvalues, first 8:")
print("    computed  " + " ".join(f"{v:10.5f}" for v in ev))
print("    (2j)^2    " + " ".join(f"{(2*(j+1))**2:10.5f}" for j in range(8)))
print("    sqrt      " + " ".join(f"{np.sqrt(v):10.5f}" for v in ev))
print("    lam_+ = 2j - mu  " + " ".join(f"{np.sqrt(v)-mu:+8.4f}" for v in ev[:6]))
print("    lam_- = -2j - mu " + " ".join(f"{-np.sqrt(v)-mu:+8.4f}" for v in ev[:6]))
print(f"\n  => {(np.sqrt(ev) - mu > 0).sum()}/8 of the '+' branches GROW. "
      f"slowest growth e^{{{np.sqrt(ev[0])-mu:+.4f} xi}} = r^{{{np.sqrt(ev[0])-mu:+.4f}}}")

# --- now look at the ACTUAL converged tail -------------------------------
St, x, r, cl, cw = pst.converge_exact(N)
Ot, Bt = St.S.unpack(x[:-2])
Pt = St.C.poisson(Ot)
xi = St.C.x
jm = St.C.nb // 2
print(f"\n  converged: ||F||={r:.2e} alpha={cw/cl:+.8f}   mid-beta column (b={St.C.b[jm]:.4f})")
print(f"  {'xi':>8} {'r':>12} {'Ot':>12} {'Bt':>12} {'Pt':>12} {'dPt/dxi':>12}")
Pt_x = St.C.dx(Pt)
sel = [k for k in range(len(xi)) if xi[k] > 8.0]
for k in sel[::max(1, len(sel) // 12)]:
    print(f"  {xi[k]:8.3f} {np.exp(xi[k])-1:12.3e} {Ot[k,jm]:12.6f} {Bt[k,jm]:12.6f} "
          f"{Pt[k,jm]:12.6f} {Pt_x[k,jm]:12.4e}")
print(f"  {xi[-1]:8.3f} {np.exp(xi[-1])-1:12.3e} {Ot[-1,jm]:12.6f} {Bt[-1,jm]:12.6f} "
      f"{Pt[-1,jm]:12.6f} {Pt_x[-1,jm]:12.4e}   <- d_xi Pt PINNED to 0 here")

# --- fit the tail decay rate of Ot: theory says e^{a0 xi} = r^{a0} --------
mask = (xi > 10) & (xi < 20)
for nm, F in (("Ot", Ot), ("Bt", Bt), ("Pt", Pt)):
    v = F[mask, jm]
    inf_ = F[-1, jm]
    d = v - v[-1]
    ok = np.abs(d) > 1e-14
    if ok.sum() > 3:
        p = np.polyfit(xi[mask][ok], np.log(np.abs(d[ok])), 1)
        print(f"  tail  {nm}: log|F - F(20)| slope = {p[0]:+.5f}   "
              f"(a0 = {a0:+.5f},  2-mu = {2-mu:+.5f})")
