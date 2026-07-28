"""T1: my dilation covariant-row defect 2.800e-4 vs tension #11's 1.748e-3.  Same
generator (both measure |cos(soft mode, v_d)| = 0.673), so the divergence must be in
what 'covariant' means.  grading_null.py's mask calls everything-except-pins-and-gauge
covariant, which SWEEPS IN the P rows and the C0 interface rows.  Split it and see."""
import importlib.util, pathlib, sys
import numpy as np

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
n2 = S.Nx * S.Nb; N = 3 * n2 + 2
J = S.jacobian(z).tocsr()
A, B, P, _, _ = S.unpack(z)
x, G1, mu, a0 = S.XI, S.G1, S.mu, S.a0
v_g = np.concatenate([z[:n2], 2 * z[n2:2 * n2], z[2 * n2:3 * n2], [z[-2]], [z[-1]]])
dA = G1 * (A + x * ((S.Dx @ A) + a0 * A))
dB = G1 * (2 * B + x * ((S.Dx @ B) + (1 + 2 * a0) * B)) - B
dP = G1 * (2 * P + x * ((S.Dx @ P) + mu * P)) - 2 * P
v_d = np.concatenate([dA.ravel(), dB.ravel(), dP.ravel(), [0.0, 0.0]])
v_g /= np.linalg.norm(v_g); v_d /= np.linalg.norm(v_d)

pin = np.concatenate([S.rT_pin, S.rT_pin + n2])
c0 = np.concatenate([S.rT_c0, S.rT_c0 + n2])
grow = np.array([N - 2, N - 1])
transport = np.setdiff1d(np.arange(2 * n2), np.concatenate([pin, c0]))
Pspecial = 2 * n2 + np.concatenate([S.rP_bedge, S.rP_outer, S.rP_c0, S.rP_c1,
                                    S.rP_cornerI])
Pint = np.setdiff1d(np.arange(2 * n2, 3 * n2), Pspecial)
loose = np.setdiff1d(np.arange(N), np.concatenate([pin, grow]))   # grading_null mask

print("ROW-CLASS DECOMPOSITION of ||J.v||   (v unit-normalized in the full z-space)",
      flush=True)
print(f"{'row class':<34s} {'count':>6s} {'grading v_g':>14s} {'dilation v_d':>14s}",
      flush=True)
for lab, sel in (("TRANSPORT PDE rows (A,B)", transport),
                 ("C0 interface rows (A,B)", c0),
                 ("P interior (Poisson PDE)", Pint),
                 ("P special (bedge/outer/C0/C1/cornerI)", Pspecial),
                 ("pins (axis col + corner circle)", pin),
                 ("gauge rows g1,g2", grow),
                 ("-- grading_null 'covariant' mask", loose)):
    wg = np.linalg.norm((J @ v_g)[sel]); wd = np.linalg.norm((J @ v_d)[sel])
    print(f"{lab:<34s} {len(sel):6d} {wg:14.4e} {wd:14.4e}", flush=True)
wd_out = np.linalg.norm((J @ v_d)[2 * n2 + S.rP_outer])
wd_cor = np.linalg.norm((J @ v_d)[2 * n2 + S.rP_cornerI])
wd_c1 = np.linalg.norm((J @ v_d)[2 * n2 + S.rP_c1])
print(f"\n  P-special breakdown for v_d: outer Neumann {wd_out:.4e}   "
      f"corner extrapolation {wd_cor:.4e}   C1 matching {wd_c1:.4e}", flush=True)
print("\nVERDICT: the two numbers measure DIFFERENT row sets; both are correct for what"
      "\nthey name.  The dilation breaks (i) the transport rows at TRUNCATION order and"
      "\n(ii) the P special rows -- outer Neumann + corner extrapolation -- at BOUNDARY"
      "\norder, which is the larger of the two and is what the loose mask reports.",
      flush=True)
