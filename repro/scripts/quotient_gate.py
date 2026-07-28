"""Q2 QUOTIENT: build the projector, measure sigma_min before and after.

THE REDUCTION.  State space  W = free field coordinates = {(dA_r, dB_r) : r in liveT}.
The pinned rows (axis column + corner circle) and the C0 interface rows are ALGEBRAIC
restrictions on the state: dA_r = 0 on pins, dA_left = dA_right on interfaces.  dP is
slaved by the linear Poisson block; (c_l, c_w) are slaved by the two gauge rows
(index-2 Hessenberg):

    xdot = M x + Bc c ,  0 = Cg x   ==>  c = -(CgBc)^-1 Cg M x ,
    xdot = [I - Bc (CgBc)^-1 Cg] M x  =:  L x .

WHY dA = 0 ON THE CORNER CIRCLE IS *NOT* AN EXTRA ASSUMPTION.  Corner regularity forces
Om = w_x(0) y1 + O(r^2), i.e. A(0,b) = w_x(0) cos b EXACTLY -- one scalar, not Nb of
them.  So an admissible perturbation has dA(0,b) = dw_x cos b, and the gauge row g1
(continuum value: A(0,b0) - WX) sets dw_x = 0.  Same for B with cos^2 and THXX.  The
corner pin and the gauge are the SAME condition in the continuum, discretely
independent (the solver's own note) -- which is itself a candidate explanation for the
near-null of J that has nothing to do with any symmetry.

WHAT THIS SCRIPT MEASURES, all of it falsifiable:
  [a] how much of the grading generator v1 lives on the PINNED nodes (it is excluded
      from the state space by that much, before any gauge is applied);
  [b] whether v1 restricted to the state space is still a soft direction of L;
  [c] the sigma_min ladder: full J -> reduced L on ker(Cg) -> L on ker(Cg) minus the
      grading shadow.  If rungs 2 and 3 agree, the DAE REDUCTION *IS* the quotient and
      the explicit deflation buys nothing;
  [d] gate G1 literally (||P v||/||v|| < 1e-10) for both projectors;
  [e] where the soft mode of J actually lives (radially), which discriminates
      "symmetry" from "pin/gauge near-redundancy at the corner".
"""
import importlib.util, pathlib, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)

d = np.load(SCR / "hunt_fields/rung_00_a-0.344712.npz")
a, z = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12), Nb=36,
                       eps_b=1e-4, alpha=a)
S.adopt_seed(z)
Nx, Nb = S.Nx, S.Nb; n2 = Nx * Nb
J = S.jacobian(z).tocsr(); Jc = J.tocsc()
A, B, P, cl, cw = S.unpack(z)
st = np.load(SCR / "realization_state.npz")
liveT = st["liveT"]; v1 = st["v1"]; v2 = st["v2"]; soft = st["softmode"]
nf = 2 * liveT.size
print(f"state dim n_f = {nf}   full J dim = {J.shape[0]}", flush=True)

free_rows = np.concatenate([liveT, n2 + liveT])
part_l = np.array(sorted(S.rT_c0), dtype=int)
part_r = np.array([S.partner[int(r)] for r in part_l], dtype=int)
spec_p = np.concatenate([S.rP_bedge, S.rP_outer, S.rP_c0, S.rP_c1, S.rP_cornerI])
coefP = np.broadcast_to(-(S.XI * S.G1 ** 2), (Nx, Nb)).ravel().copy()
coefP[spec_p] = 0.0

def embed(x):
    full = np.zeros(J.shape[1], dtype=x.dtype)
    full[free_rows] = x
    full[part_l] = full[part_r]
    full[n2 + part_l] = full[n2 + part_r]
    full[2 * n2:3 * n2] = S._lp_factor().solve(coefP * full[:n2])
    return full

def embedT(w):
    """adjoint of embed."""
    q = S._lp_factor().solve(np.asarray(w[2 * n2:3 * n2]), trans="T")
    out = np.zeros(J.shape[1], dtype=w.dtype)
    out[:n2] = w[:n2] + coefP * q
    out[n2:2 * n2] = w[n2:2 * n2]
    out[part_r] += out[part_l]; out[n2 + part_r] += out[n2 + part_l]
    return out[free_rows]

Bc = np.asarray(J[:, -2:].todense())[free_rows, :]
Cg = np.asarray(J[-2:, :].todense())[:, free_rows]
CgB = Cg @ Bc; CgBi = np.linalg.inv(CgB)
print(f"Cg Bc = {CgB.ravel()}   cond={np.linalg.cond(CgB):.3e}", flush=True)
Jf = J[free_rows, :]

def Mx(x):  return Jf @ embed(x)
def MxT(y): return embedT(Jf.T @ y)
def Lx(x):
    y = Mx(x); return y - Bc @ (CgBi @ (Cg @ y))
def LxT(y):
    return MxT(y - Cg.T @ (CgBi.T @ (Bc.T @ y)))

lu = spla.splu(Jc); luH = spla.splu(Jc.T.tocsc())
def Linv(f):
    rhs = np.zeros(J.shape[0], dtype=f.dtype); rhs[free_rows] = f
    return lu.solve(rhs)[free_rows]
def LinvH(f):
    rhs = np.zeros(J.shape[0], dtype=f.dtype); rhs[free_rows] = f
    return luH.solve(rhs)[free_rows]

Qc, _ = np.linalg.qr(Cg.T)
def Pker(x): return x - Qc @ (Qc.T @ x)

# ---------------------------------------------------------------------------
# [a] how much of the grading generator is EXCLUDED by the pins alone
# ---------------------------------------------------------------------------
fieldslice = np.concatenate([np.arange(n2), n2 + np.arange(n2)])
v1_field = v1[fieldslice]; v2_field = v2[fieldslice]
def pinmass(vv):
    keep = np.linalg.norm(vv[free_rows]); tot = np.linalg.norm(vv[fieldslice])
    return 1.0 - keep / tot, keep / tot
m1, k1 = pinmass(v1); m2, k2 = pinmass(v2)
print(f"\n[a] EXCLUDED BY THE PINS/C0 ALONE (no gauge yet):", flush=True)
print(f"    grading  v1: {m1:.4%} of its field norm sits on pinned/duplicate nodes",
      flush=True)
print(f"    dilation v2: {m2:.4%}", flush=True)
cor = np.array([j for j in range(Nb)])
print(f"    v1 on the CORNER circle alone: "
      f"{np.linalg.norm(v1[cor])/np.linalg.norm(v1[fieldslice]):.4%}   "
      f"on the AXIS column: "
      f"{np.linalg.norm(v1[[i*Nb+Nb-1 for i in range(Nx)]])/np.linalg.norm(v1[fieldslice]):.4%}",
      flush=True)

v1f = v1[free_rows] / np.linalg.norm(v1[free_rows])
v2f = v2[free_rows] / np.linalg.norm(v2[free_rows])
softf = soft[free_rows] / np.linalg.norm(soft[free_rows])
sin1 = np.linalg.norm(v1f - Pker(v1f)); sin2 = np.linalg.norm(v2f - Pker(v2f))
print(f"    then sin(angle(v1_f, ker Cg)) = {sin1:.4e}   sin(v2_f) = {sin2:.4e}",
      flush=True)

# ---------------------------------------------------------------------------
# [b] is the truncated grading direction still soft for L?
# ---------------------------------------------------------------------------
u = np.random.default_rng(11).standard_normal(nf); u /= np.linalg.norm(u)
vv = np.random.default_rng(12).standard_normal(nf); vv /= np.linalg.norm(vv)
lhs = float(Lx(u) @ vv); rhs_ = float(u @ LxT(vv))
print(f"\n[b] adjoint check  <Lu,v>={lhs:.10e}  <u,L^Tv>={rhs_:.10e}  "
      f"rel={abs(lhs-rhs_)/max(abs(lhs),1e-300):.2e}", flush=True)
print(f"    ||M v1_f||={np.linalg.norm(Mx(v1f)):.4e}   ||L v1_f||={np.linalg.norm(Lx(v1f)):.4e}",
      flush=True)
print(f"    ||L v2_f||={np.linalg.norm(Lx(v2f)):.4e}   "
      f"||L x_rand||={np.linalg.norm(Lx(u)):.4e}", flush=True)

def opnorm(op, opH, proj, n, iters=150, seed=0):
    x = np.random.default_rng(seed).standard_normal(n); x = proj(x); x /= np.linalg.norm(x)
    s = 0.0
    for _ in range(iters):
        y = proj(op(x)); x2 = proj(opH(y)); nn = np.linalg.norm(x2)
        if nn == 0: return 0.0, x
        x = x2 / nn; s = np.sqrt(nn)
    return s, x
nrmL, _ = opnorm(Lx, LxT, Pker, nf, iters=200, seed=1)
print(f"    ||L||_2 on ker(Cg) = {nrmL:.4e}   <- TRUE OPERATOR SCALE", flush=True)

xk = Pker(np.random.default_rng(13).standard_normal(nf)); xk /= np.linalg.norm(xk)
print(f"    [route agreement] ||L^-1 L x - x||/||x|| = "
      f"{np.linalg.norm(Linv(Lx(xk))-xk):.3e}", flush=True)

# ---------------------------------------------------------------------------
# [c] the sigma_min ladder
# ---------------------------------------------------------------------------
w1 = Pker(v1f); w1 /= np.linalg.norm(w1)
def Pquot(x):
    y = Pker(x); return y - w1 * float(w1 @ y)
t2 = Pquot(v2f); w2 = t2 / np.linalg.norm(t2)
def Pquot2(x):
    y = Pquot(x); return y - w2 * float(w2 @ y)

print(f"\n[c] SIGMA_MIN LADDER (sigma_min = 1/||L^-1|| on the stated space):", flush=True)
res = {}
for tag, proj, seed in (("ker(Cg) only        [BEFORE]", Pker, 2),
                        ("ker(Cg) - grading   [AFTER ]", Pquot, 3),
                        ("ker(Cg) - grad - dilation   ", Pquot2, 4)):
    ni, xv = opnorm(Linv, LinvH, proj, nf, iters=250, seed=seed)
    res[tag] = 1.0 / ni
    print(f"    {tag}  ||L^-1||={ni:.4e}   sigma_min={1.0/ni:.4e}   "
          f"|cos(argmin, v1)|={abs(float(xv@v1f)):.4f}", flush=True)
s_ker = res["ker(Cg) only        [BEFORE]"]; s_q1 = res["ker(Cg) - grading   [AFTER ]"]
print(f"    sigma_min(J_full) (prerequisite measurement) = {float(st['smin']):.4e}",
      flush=True)
print(f"    RATIO sigma_min(L|ker Cg) / sigma_min(J_full) = {s_ker/float(st['smin']):.4e}",
      flush=True)
print(f"    RATIO sigma_min(L|ker Cg) / ||L||             = {s_ker/nrmL:.4e}", flush=True)

# ---------------------------------------------------------------------------
# [d] gate G1, literally, for both projectors
# ---------------------------------------------------------------------------
print(f"\n[d] GATE G1  ||P v1||/||v1||   (need < 1e-10)", flush=True)
print(f"    P = restriction to ker(Cg) (structural quotient) : "
      f"{np.linalg.norm(Pker(v1f)):.6e}", flush=True)
print(f"    P = ker(Cg) minus the grading shadow (explicit)  : "
      f"{np.linalg.norm(Pquot(v1f)):.3e}", flush=True)
print(f"    (on the FULL v1, incl. pinned nodes: structural  : "
      f"{np.linalg.norm(Pker(v1[free_rows]))/np.linalg.norm(v1):.6e})", flush=True)
leak = np.linalg.norm(Pquot(LxT(w1)))
print(f"    LEAKAGE ||P_V L^T w1|| = {leak:.4e}  (/||L|| = {leak/nrmL:.3e}) "
      f"-> V is L-invariant only to this", flush=True)

# ---------------------------------------------------------------------------
# [e] where does the soft mode of J live?
# ---------------------------------------------------------------------------
sm = soft.copy()
blocks = {"A": sm[:n2], "B": sm[n2:2*n2], "P": sm[2*n2:3*n2],
          "c": sm[3*n2:]}
tot = np.linalg.norm(sm)
print(f"\n[e] SOFT MODE OF J -- block mass: " +
      "  ".join(f"{k}={np.linalg.norm(v)/tot:.4f}" for k, v in blocks.items()), flush=True)
Amode = sm[:n2].reshape(Nx, Nb)
rad = np.linalg.norm(Amode, axis=1)
order = np.argsort(-rad)[:6]
print(f"    top radial nodes (i, xi, mass frac): " +
      "  ".join(f"({int(i)}, {S.x[int(i)]:.4g}, {rad[int(i)]/np.linalg.norm(Amode):.3f})"
                for i in order), flush=True)
print(f"    corner-circle mass frac of the A block = "
      f"{rad[0]/np.linalg.norm(Amode):.4f}   (1 node of {Nx})", flush=True)
Gm = np.array([[1.0, float(v1f @ v2f)], [float(v1f @ v2f), 1.0]])
bb = np.array([float(softf @ v1f), float(softf @ v2f)])
cc = np.linalg.solve(Gm, bb)
print(f"    soft mode restricted to the state space: |cos v1_f|={bb[0]:.4f} "
      f"|cos v2_f|={bb[1]:.4f}  ||proj span{{v1,v2}}||={np.sqrt(max(bb@cc,0)):.4f}",
      flush=True)
np.savez(SCR / "quotient_state.npz", w1=w1, w2=w2, Qc=Qc, free_rows=free_rows,
         s_ker=s_ker, s_q1=s_q1, nrmL=nrmL, sin1=sin1, m1=m1)
print("saved quotient_state.npz", flush=True)
