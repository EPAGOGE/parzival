"""TENSION #6 + spectrum gate 0: is the measured soft mode the exact scaling symmetry?

The symmetry theorem (axiom cornerreg_exact_scaling_symmetry) says the covariant rows
are homogeneous of degree 2/3/1 under A->sA, B->s^2 B, P->sP, cl->s cl, cw->s cw.
Euler: for a row R of degree d, DR.v = d*R, where v is the infinitesimal generator
v = (A, 2B, P, cl, cw).  AT A ROOT R = 0, so J.v = 0 EXACTLY on every covariant row --
the symmetry is a structural near-null of the Jacobian, broken ONLY by the static pins
and gauge rows.  Two consequences, both measured here:
  (a) tension #6: does the soft mode carry the cl,cw components the theorem predicts
      (dcl/cl = dcw/cw = ds), or is it the gauge-projected shadow?
  (b) spectrum prerequisite: this direction must be quotiented out before any
      resolvent/pseudospectrum computation, or it contaminates the smallest modes."""
import importlib.util, pathlib, sys
import numpy as np
import scipy.sparse.linalg as spla
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
spec = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(spec); sys.modules["pc"] = pc; spec.loader.exec_module(pc)

d = np.load(SCR / "hunt_fields/rung_00_a-0.344712.npz")
a, z = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0,2.0,15.0,25.0), degs=(16,40,12), Nb=36, eps_b=1e-4, alpha=a)
S.adopt_seed(z)
n2 = S.Nx * S.Nb
F = S.residual(z); J = S.jacobian(z).tocsr()
print(f"ground root: alpha={a:+.8f}  ||F||_rms={np.linalg.norm(F)/np.sqrt(F.size):.2e}  n={z.size}", flush=True)

# --- the grading generator v = (A, 2B, P, cl, cw) ---------------------------
v = np.concatenate([z[:n2], 2.0*z[n2:2*n2], z[2*n2:3*n2], [z[-2]], [z[-1]]])
v /= np.linalg.norm(v)
w = J @ v
Jn = spla.norm(J) if hasattr(spla, "norm") else np.sqrt((J.multiply(J)).sum())
print(f"\n[A] EULER TEST  ||J.v|| / ||J|| = {np.linalg.norm(w)/Jn:.3e}   (raw ||J.v||={np.linalg.norm(w):.3e})", flush=True)
# which rows carry it?
pin = np.concatenate([S.rT_pin, S.rT_pin + n2])          # transport pins (A and B blocks)
gauge = np.array([J.shape[0]-2, J.shape[0]-1])
mask = np.zeros(J.shape[0], bool); mask[pin] = True; mask[gauge] = True
print(f"    carried by pins+gauge rows: {np.linalg.norm(w[mask])/max(np.linalg.norm(w),1e-300):.4%}"
      f"   by all other rows: {np.linalg.norm(w[~mask])/max(np.linalg.norm(w),1e-300):.4%}", flush=True)
print(f"    covariant-row residual ||J.v||_cov = {np.linalg.norm(w[~mask]):.3e}  (theorem: 0)", flush=True)

# --- smallest singular direction of J ---------------------------------------
lu = spla.splu(J.tocsc()); luT = spla.splu(J.T.tocsc())
x = np.random.default_rng(3).standard_normal(z.size); x /= np.linalg.norm(x)
for _ in range(60):
    x = lu.solve(luT.solve(x)); x /= np.linalg.norm(x)
smin = np.linalg.norm(J @ x)
print(f"\n[B] SOFT MODE   sigma_min = {smin:.3e}", flush=True)
cosang = abs(float(x @ v))
print(f"    |cos(soft mode, grading generator)| = {cosang:.6f}", flush=True)

# --- tension #6: does the soft mode carry the predicted cl,cw components? ----
# theorem: along the exact mode, dcl/cl = dcw/cw = ds (same relative motion)
s_field = float(x[:n2] @ z[:n2]) / float(z[:n2] @ z[:n2])      # implied ds from A block
print(f"\n[C] TENSION #6 -- cl,cw components of the soft mode", flush=True)
print(f"    implied ds from A block          = {s_field:+.6e}", flush=True)
print(f"    dcl/cl (measured on soft mode)   = {x[-2]/z[-2]:+.6e}", flush=True)
print(f"    dcw/cw (measured on soft mode)   = {x[-1]/z[-1]:+.6e}", flush=True)
r_cl = (x[-2]/z[-2]) / s_field if s_field else float('nan')
r_cw = (x[-1]/z[-1]) / s_field if s_field else float('nan')
print(f"    ratio to prediction (theorem=1):   cl {r_cl:+.4f}   cw {r_cw:+.4f}", flush=True)
verdict = ("EXACT MODE (cl,cw move with the fields as the theorem predicts)"
           if min(abs(r_cl), abs(r_cw)) > 0.5 else
           "GAUGE-PROJECTED SHADOW (pins/gauge rows suppress the cl,cw components)")
print(f"    VERDICT: {verdict}", flush=True)
print("done", flush=True)
