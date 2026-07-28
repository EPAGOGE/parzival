"""Q1 REALIZATION + Q2 NULL INVENTORY, measured on the live ground root.

Derivation carried by this script (numbers, not assertions):

  R1  The DESCRIPTOR PENCIL.  polar_march.py's docstring states the dynamic-rescaling
      evolution in the SAME normalization the steady residual is written in:
          d_t Ot = (c_w - c_l a0) Ot - c_l Ot_s - e^(a0 s)[bracket] + e^(a0 s)[source]
      i.e. the coefficient of d_t Ot is EXACTLY the coefficient of the undifferentiated
      c_w Ot term, namely 1.  The corner regularization multiplies BOTH the state
      (Ot = xi A) and the row (RO' = RO/xi) by the same time-independent power, so
          d_tau A = RO' ,   d_tau B = RB'
      with coefficient exactly 1 and NO weight.  Therefore E is a 0/1 DIAGONAL MASK:
      1 on the live transport rows, 0 on the pinned/C0/Poisson/gauge rows.
      Checked here by C1: the c_w column of J restricted to the live A rows must equal
      A itself (that IS the mass operator), and the c_w column on the live B rows must
      equal 2B.

  R2  THE TWO EXACT CONTINUUM SYMMETRIES and their generators in the divided variables.
      (S1) grading   Om->s Om, Bt->s^2 Bt, Psi->s Psi, c->s c
           v1 = (A, 2B, P, c_l, c_w)                            [already measured]
      (S2) dilation  Om_L(y)=Om(Ly), B_L=L^-1 B(Ly), Psi_L=L^-2 Psi(Ly), c UNCHANGED
           generator in physical fields: (r d_r Om, -B + r d_r B, -2 Psi + r d_r Psi).
           Pushed through Om = e^{a0 xi} xi A, B = e^{(1+2a0) xi} xi^2 B,
           Psi = e^{mu xi} xi^2 P  with r d_r = g d_xi, g = xi G1:
               dA = G1 (LA A)
               dB = -B + G1 (LB2 B)
               dP = -2P + G1 (LPmu P)
               dc_l = dc_w = 0
           EXACTLY the operator bundles the solver already assembles.  Euler measured.
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
Nx, Nb = S.Nx, S.Nb
n2 = Nx * Nb
F = S.residual(z)
J = S.jacobian(z).tocsr()
A, B, P, cl, cw = S.unpack(z)
print(f"ROOT  alpha={a:+.8f}  Nx={Nx} Nb={Nb}  n2={n2}  n={z.size}", flush=True)
print(f"      ||F||_rms={np.linalg.norm(F)/np.sqrt(F.size):.3e}  c_l={cl:.8f}  "
      f"c_w={cw:.8f}  h_id={S.h_id(z):+.3e}", flush=True)

# ---------------------------------------------------------------------------
# ROW PARTITION -> the mass matrix E
# ---------------------------------------------------------------------------
allrows = np.arange(n2)
algT = np.union1d(S.rT_pin, S.rT_c0)                       # algebraic transport rows
liveT = np.setdiff1d(allrows, algT)                        # live rows in EACH of A,B
algP = np.unique(np.concatenate([S.rP_bedge, S.rP_outer, S.rP_c0,
                                 S.rP_c1, S.rP_cornerI]))
liveP = np.setdiff1d(allrows, algP)
nrow = J.shape[0]
mask = np.zeros(nrow)
mask[liveT] = 1.0                       # A block
mask[n2 + liveT] = 1.0                  # B block
E = sp.diags(mask, format="csr")
print(f"\n[R1] PENCIL  (E, J):  E = diag 0/1,  rank(E) = {int(mask.sum())} "
      f"of {nrow}", flush=True)
print(f"     rows: liveT/block={liveT.size}  pins={S.rT_pin.size}  C0={S.rT_c0.size}"
      f"   |  P: live={liveP.size} alg={algP.size}  |  gauge=2", flush=True)

# C1: the c_w column IS the mass operator on the live rows.
Bc = np.asarray(J[:, -2:].todense())
cw_col = Bc[:, 1]
eA = np.linalg.norm(cw_col[liveT] - A.ravel()[liveT]) / np.linalg.norm(A.ravel()[liveT])
eB = (np.linalg.norm(cw_col[n2 + liveT] - 2.0 * B.ravel()[liveT])
      / np.linalg.norm(2.0 * B.ravel()[liveT]))
print(f"[C1] d(RO')/dc_w  vs  A   rel err = {eA:.3e}      (0 => d_tau A carries "
      f"coefficient 1)", flush=True)
print(f"     d(RB')/dc_w  vs  2B  rel err = {eB:.3e}      (0 => d_tau B carries "
      f"coefficient 1)", flush=True)
print(f"     c_w column on ALGEBRAIC rows: ||.||={np.linalg.norm(cw_col[algT]):.3e} "
      f"(A) {np.linalg.norm(cw_col[n2+algT]):.3e} (B)  "
      f"{np.linalg.norm(cw_col[2*n2:]):.3e} (P+gauge)", flush=True)

# ---------------------------------------------------------------------------
# ROW SCALES -- the pencil is block-scaled; this is a real pitfall
# ---------------------------------------------------------------------------
rn = np.sqrt(np.asarray(J.multiply(J).sum(axis=1)).ravel())
def rep(name, idx):
    v = rn[idx]
    print(f"     {name:<16s} n={len(idx):5d}  row-norm  min={v.min():.3e} "
          f"med={np.median(v):.3e} max={v.max():.3e}", flush=True)
print(f"\n[R1b] ROW SCALES of J (||J||_F={spla.norm(J):.3e}):", flush=True)
rep("liveT A", liveT); rep("liveT B", n2 + liveT)
rep("pins A", S.rT_pin); rep("P live", 2 * n2 + liveP); rep("P alg", 2 * n2 + algP)
rep("gauge", np.array([nrow - 2, nrow - 1]))

# ---------------------------------------------------------------------------
# [R2] EULER TESTS for the two exact symmetries
# ---------------------------------------------------------------------------
pinmask = np.zeros(nrow, bool)
pinmask[S.rT_pin] = True; pinmask[n2 + S.rT_pin] = True
pinmask[S.rT_c0] = True; pinmask[n2 + S.rT_c0] = True
pinmask[-2:] = True                                 # gauge
covmask = np.zeros(nrow, bool)
covmask[liveT] = True; covmask[n2 + liveT] = True; covmask[2 * n2 + liveP] = True
pmask = np.zeros(nrow, bool); pmask[2 * n2 + algP] = True   # P algebraic rows

Jn = spla.norm(J)

def euler(name, v):
    v = v / np.linalg.norm(v)
    w = J @ v
    print(f"\n[R2:{name}]  ||J.v||={np.linalg.norm(w):.3e}   /||J||={np.linalg.norm(w)/Jn:.3e}",
          flush=True)
    print(f"     covariant rows (liveT + liveP): {np.linalg.norm(w[covmask]):.3e}", flush=True)
    print(f"     pins + C0 + gauge rows        : {np.linalg.norm(w[pinmask]):.3e}", flush=True)
    print(f"     P algebraic rows              : {np.linalg.norm(w[pmask]):.3e}", flush=True)
    return v, w

v1 = np.concatenate([A.ravel(), 2.0 * B.ravel(), P.ravel(), [cl], [cw]])
v1, w1 = euler("S1 grading ", v1)

G1 = S.G1; xi = S.XI; a0 = S.a0; mu = S.mu
LA_A = A + xi * ((S.Dx @ A) + a0 * A)
LB2_B = 2.0 * B + xi * ((S.Dx @ B) + (1.0 + 2.0 * a0) * B)
LP_P = 2.0 * P + xi * ((S.Dx @ P) + mu * P)
dA = G1 * LA_A
dB = -B + G1 * LB2_B
dP = -2.0 * P + G1 * LP_P
v2 = np.concatenate([dA.ravel(), dB.ravel(), dP.ravel(), [0.0], [0.0]])
v2, w2 = euler("S2 dilation", v2)
# split the dilation defect by sub-row-class to see WHAT breaks it
for nm, idx in (("liveT A", liveT), ("liveT B", n2 + liveT), ("P live", 2 * n2 + liveP),
                ("P outer", 2 * n2 + S.rP_outer), ("P cornerI", 2 * n2 + S.rP_cornerI),
                ("P C1", 2 * n2 + S.rP_c1), ("P bedge", 2 * n2 + S.rP_bedge)):
    print(f"       {nm:<10s} {np.linalg.norm(w2[idx]):.3e}", flush=True)
print(f"     |cos(v1,v2)| = {abs(float(v1 @ v2)):.6f}", flush=True)

# ---------------------------------------------------------------------------
# sigma_min of the FULL J (reproduce the prerequisite measurement)
# ---------------------------------------------------------------------------
lu = spla.splu(J.tocsc()); luT = spla.splu(J.T.tocsc())
rng = np.random.default_rng(3)
x = rng.standard_normal(z.size); x /= np.linalg.norm(x)
for _ in range(80):
    x = lu.solve(luT.solve(x)); x /= np.linalg.norm(x)
smin = np.linalg.norm(J @ x)
print(f"\n[B] sigma_min(J_full) = {smin:.4e}   |cos(soft, v1)|={abs(float(x@v1)):.6f}"
      f"   |cos(soft, v2)|={abs(float(x@v2)):.6f}", flush=True)
np.savez(SCR / "realization_state.npz", v1=v1, v2=v2, softmode=x, smin=smin,
         liveT=liveT, algT=algT, liveP=liveP, algP=algP, mask=mask)
print("saved realization_state.npz", flush=True)
