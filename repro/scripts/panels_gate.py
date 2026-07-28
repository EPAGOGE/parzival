"""GATES for polar_panels.py, in the order they can fail.

G0  JACOBIAN EXACTNESS. FD-check random field columns AND both c-columns of the sparse
    Jacobian at a perturbed point, for a K=2 panel grid (interfaces exercised).  The
    entire value of the solver rests on this matrix being the true derivative; a wrong
    block converges Newton to a tiny ||F|| at the WRONG point -- the classic failure
    class of this project.
G1  REPRODUCTION. K=1 single panel [0,15], degree 35 (36 nodes) is the SAME discrete
    system as the single-grid solver at N=36/XMAX=15/d1/eps_b=1e-3 -- same nodes, same
    beta grid, same constraints; only Pt-elimination differs, which cannot move the root.
    Known value from the collapse-test data: alpha = -0.33840790.  Agreement to ~1e-7
    or the new solver is wrong somewhere the FD check cannot see (row bookkeeping).
G2  FIRST PANEL RUN. edges [0,2,15], degs (16,48): 65 radial nodes, mid-band getting 3x
    the nodes the global-36 grid gave it, at a Jacobian size the sparse LU eats easily.
    Report alpha + interface jumps (C0/C1 residuals must sit at solver tolerance).
"""
import importlib.util, pathlib, sys, time
import numpy as np
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")


def mod(n, f):
    sp_ = importlib.util.spec_from_file_location(n, str(H / f))
    m = importlib.util.module_from_spec(sp_); sys.modules[n] = m
    sp_.loader.exec_module(m); return m


pp = mod("pp", "polar_panels.py")
REF = -0.34240009

print("G0  JACOBIAN FD CHECK (K=2, interfaces live)", flush=True)
S = pp.PanelSolver(edges=(0.0, 2.0, 15.0), degs=(10, 18), Nb=20)
z = S.pack(S.Ot0, S.Bt0, S.Pt0, S.P["cl"], S.P["cw"])
rng = np.random.default_rng(0)
z[:-2] += 1e-3 * rng.standard_normal(z.size - 2)
J = S.jacobian(z).toarray()
f0 = S.residual(z)
h = 1e-7 * max(np.linalg.norm(z), 1.0)
cols = list(rng.choice(z.size - 2, size=8, replace=False)) + [z.size - 2, z.size - 1]
worst = 0.0
for j in cols:
    e = np.zeros(z.size); e[j] = h
    num = (S.residual(z + e) - S.residual(z - e)) / (2 * h)
    err = np.linalg.norm(num - J[:, j]) / max(np.linalg.norm(num), 1e-300)
    kind = "c_l" if j == z.size - 2 else ("c_w" if j == z.size - 1 else "field")
    print(f"    col {j:6d} [{kind:5s}] rel err {err:.3e}", flush=True)
    worst = max(worst, err)
print(f"    worst = {worst:.3e}  -> {'EXACT' if worst < 1e-5 else 'MISMATCH -- STOP'}",
      flush=True)
if worst >= 1e-5:
    sys.exit(1)

print("\nG1  K=1 REPRODUCTION vs single-grid N=36/L=15/d1  (known alpha -0.33840790)",
      flush=True)
t0 = time.time()
S1, z1, r1, info1 = pp.converge(edges=(0.0, 15.0), degs=(35,), Nb=36)
a1 = info1.get("alpha", float("nan"))
print(f"    panels K=1: alpha={a1:+.8f}  ||F||={r1:.2e}  passes={info1.get('passes')}"
      f"  conv={info1.get('converged')}  secs={time.time()-t0:.0f}", flush=True)
print(f"    known     : alpha=-0.33840790   d(alpha)={abs(a1 - (-0.33840790)):.2e}"
      f"  -> {'PASS' if abs(a1 - (-0.33840790)) < 1e-6 else 'FAIL'}", flush=True)

print("\nG2  FIRST PANEL RUN  edges [0,2,15] degs (16,48)", flush=True)
t0 = time.time()
S2, z2, r2, info2 = pp.converge(edges=(0.0, 2.0, 15.0), degs=(16, 48), Nb=36)
a2 = info2.get("alpha", float("nan"))
print(f"    alpha={a2:+.8f} ({100*(a2-REF)/abs(REF):+.3f}% vs ref)  ||F||={r2:.2e}  "
      f"passes={info2.get('passes')}  conv={info2.get('converged')}  "
      f"secs={time.time()-t0:.0f}", flush=True)
Ot, Bt, Pt, cl, cw = S2.unpack(z2)
CLS = 2.0 * S2.THXX_REF / S2.WX_REF
print(f"    c_l={cl:.6f}  d_cl={100*(cl/CLS-1):+.2f}%", flush=True)
for k in range(1, S2.K):
    L, R = S2.lefts[k], S2.rights[k - 1]
    for nm, F in (("Ot", Ot), ("Bt", Bt), ("Pt", Pt)):
        c0 = float(np.abs(F[L, :] - F[R, :]).max())
        c1 = float(np.abs((S2.Dx @ F)[L, :] - (S2.Dx @ F)[R, :]).max())
        print(f"    iface {k} {nm}: |C0|={c0:.2e}  |C1 jump|={c1:.2e}", flush=True)
print(f"\n  reference alpha = {REF}", flush=True)
