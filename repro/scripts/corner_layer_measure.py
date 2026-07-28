"""Corner/layer measurer: candidate root (a=alpha_1) vs ground root.

Measurement only -- loads npz fields, no Newton solves.
Tasks:
  1. cl, cw, corner Poisson amplitude c=(cl-2cw)/4, cw/cl for both roots.
  2. Fit P(0,:) to c*sin(2*(b-eps_b)) (and to the exact wedge harmonic
     sin(k*(b-eps_b)), k=pi/(pi/2-2eps_b)) for both roots.
  3. Eps-flatness mechanism: relative field change per panel across the
     branch1 eps ladder; corner-panel P L2 (xi<2) branch vs ground.
  4. Corner departure: radial derivative structure of A just off the corner.
"""
import sys
import importlib.util
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
HF = SCRATCH + "/hunt_fields"

spec = importlib.util.spec_from_file_location(
    "pc", "/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py")
pc = importlib.util.module_from_spec(spec)
sys.modules["pc"] = pc
spec.loader.exec_module(pc)

# one cheap solver instance just for grid/differentiation objects (no solve)
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                       Nb=36, eps_b=1e-4, alpha=-0.4168236)
Nx, Nb = S.Nx, S.Nb
n = Nx * Nb
offs = list(map(int, S.offs))          # panel offsets, e.g. [0,17,58,71]
print(f"grid: Nx={Nx} Nb={Nb} offs={offs}")
print(f"x[0:5]={S.x[:5]}")
print(f"b[0]={S.b[0]:.6e} b[-1]={S.b[-1]:.6e}")

pm = sys.modules.get("pm")  # loaded by pc


def beta_grid(eps_b):
    b, Db, Db2 = pm.grid(Nb - 1, eps_b, np.pi / 2 - eps_b)
    return b


def load(fn):
    d = np.load(f"{HF}/{fn}")
    z = d["z"]
    assert z.size == 3 * n + 2, (fn, z.size)
    out = dict(
        A=z[:n].reshape(Nx, Nb), B=z[n:2 * n].reshape(Nx, Nb),
        P=z[2 * n:3 * n].reshape(Nx, Nb),
        cl=float(z[3 * n]), cw=float(z[3 * n + 1]),
        keys=sorted(d.keys()))
    for k in ("a", "h"):
        if k in d:
            out[k] = float(d[k])
    return out


def trapw(x):
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def l2(F, rows, x, b):
    """trapezoid-weighted L2 of F over given radial rows (full beta range)."""
    wx = trapw(x[rows])
    wb = trapw(b)
    return float(np.sqrt(np.einsum("i,ij,j->", wx, F[rows, :] ** 2, wb)))


GROUND = "rung_00_a-0.344712.npz"
CAND = "find_half.npz"

g = load(GROUND)
c1 = load(CAND)
print(f"\nground file keys={g['keys']}, a={g.get('a')}")
print(f"cand   file keys={c1['keys']}, a={c1.get('a')}")

# ---------------------------------------------------------------- task 1
print("\n=== TASK 1: corner Poisson amplitude ===")
for name, d in (("ground", g), ("candidate", c1)):
    c_alg = (d["cl"] - 2.0 * d["cw"]) / 4.0
    d["c_alg"] = c_alg
    print(f"{name:9s}: cl={d['cl']:+.9f}  cw={d['cw']:+.9f}  "
          f"c=(cl-2cw)/4={c_alg:+.9f}  cw/cl={d['cw'] / d['cl']:+.9f}")
print(f"c ratio candidate/ground        = {c1['c_alg'] / g['c_alg']:+.6f}")
print(f"cw/cl diff (cand - ground)      = "
      f"{c1['cw'] / c1['cl'] - g['cw'] / g['cl']:+.6e}")

# ---------------------------------------------------------------- task 2
print("\n=== TASK 2: P(0,:) corner-row fit ===")
EPSB = 1e-4
b = S.b
k_exact = np.pi / (np.pi / 2 - 2 * EPSB)
print(f"eps_b={EPSB}  exact wedge exponent k={k_exact:.8f}  (k-2={k_exact - 2:.4e})")
for name, d in (("ground", g), ("candidate", c1)):
    P0 = d["P"][0, :]
    for lbl, kk in (("sin(2(b-eps))", 2.0), ("sin(k(b-eps))", k_exact)):
        s = np.sin(kk * (b - EPSB))
        cfit = float(np.dot(P0, s) / np.dot(s, s))
        res = float(np.linalg.norm(P0 - cfit * s) / np.linalg.norm(P0))
        if kk == 2.0:
            d["c_fit"] = cfit
            d["fit_res"] = res
        print(f"{name:9s} {lbl}: c_fit={cfit:+.9f}  "
              f"(c_alg={d['c_alg']:+.9f}, rel diff="
              f"{(cfit - d['c_alg']) / abs(d['c_alg']):+.3e})  "
              f"rel fit residual={res:.3e}")
    print(f"{name:9s} |P(0,:)|_inf = {np.max(np.abs(P0)):.6e}")

# ---------------------------------------------------------------- task 3
print("\n=== TASK 3: eps-flatness mechanism ===")
LADDER = [("branch1_eps1e-4.npz", 1e-4), ("branch1_eps5e-05.npz", 5e-5),
          ("branch1_eps3e-05.npz", 3e-5), ("branch1_eps1e-05.npz", 1e-5)]
lad = [(load(fn), eps) for fn, eps in LADDER]

from scipy.interpolate import BarycentricInterpolator

bref = beta_grid(1e-4)


def interp_beta(F, b_src, b_dst):
    out = np.empty((F.shape[0], len(b_dst)))
    for i in range(F.shape[0]):
        out[i, :] = BarycentricInterpolator(b_src, F[i, :])(b_dst)
    return out


panels = [(offs[k], offs[k + 1]) for k in range(len(offs) - 1)]
print("relative field change per panel, beta-interpolated to eps=1e-4 grid")
print("(rows: eps pair; cols: field x panel [P0=corner xi<2, P1=mid, P2=outer])")
for (d_a, eps_a), (d_b, eps_b2) in zip(lad[:-1], lad[1:]):
    ba, bb = beta_grid(eps_a), beta_grid(eps_b2)
    line = f"eps {eps_a:.0e}->{eps_b2:.0e}: "
    for fld in ("A", "B", "P"):
        Fa = interp_beta(d_a[fld], ba, bref)
        Fb = interp_beta(d_b[fld], bb, bref)
        vals = []
        for lo, hi in panels:
            num = np.linalg.norm(Fa[lo:hi] - Fb[lo:hi])
            den = np.linalg.norm(Fa[lo:hi])
            vals.append(num / den)
        line += f" {fld}[" + " ".join(f"{v:.2e}" for v in vals) + "]"
    print(line)

# ladder endpoints: total relative motion 1e-4 -> 1e-5 per panel
d_hi, d_lo = lad[0][0], lad[-1][0]
Fa_all, Fb_all = {}, {}
print("total relative motion eps 1e-4 -> 1e-5 per panel:")
for fld in ("A", "B", "P"):
    Fa = interp_beta(d_hi[fld], beta_grid(1e-4), bref)
    Fb = interp_beta(d_lo[fld], beta_grid(1e-5), bref)
    vals = [np.linalg.norm(Fa[lo:hi] - Fb[lo:hi]) / np.linalg.norm(Fa[lo:hi])
            for lo, hi in panels]
    print(f"  {fld}: " + "  ".join(f"panel{k}={v:.3e}" for k, v in enumerate(vals)))

# corner-panel P energy, branch vs ground (both eps=1e-4)
lo, hi = panels[0]
print("\ncorner-panel (xi<2) P L2, trapezoid-weighted:")
Eg = l2(g["P"], slice(lo, hi), S.x, S.b)
Ec = l2(c1["P"], slice(lo, hi), S.x, S.b)
Eg_tot = l2(g["P"], slice(0, Nx), S.x, S.b)
Ec_tot = l2(c1["P"], slice(0, Nx), S.x, S.b)
print(f"  ground   : corner={Eg:.6e}  total={Eg_tot:.6e}  corner/total={Eg / Eg_tot:.4e}")
print(f"  candidate: corner={Ec:.6e}  total={Ec_tot:.6e}  corner/total={Ec / Ec_tot:.4e}")
print(f"  RATIO candidate/ground corner P L2 = {Ec / Eg:.6f}")
# same for nodal RMS as a robustness cross-check
rg = float(np.sqrt(np.mean(g["P"][lo:hi] ** 2)))
rc = float(np.sqrt(np.mean(c1["P"][lo:hi] ** 2)))
print(f"  nodal-RMS cross-check ratio        = {rc / rg:.6f}")
# also branch root from ladder (its own file) vs ground
rb = float(np.sqrt(np.mean(lad[0][0]["P"][lo:hi] ** 2)))
print(f"  (branch1_eps1e-4 corner P RMS)/(ground) = {rb / rg:.6f}")

# corner-panel A and B energies for context
for fld in ("A", "B"):
    Egf = l2(g[fld], slice(lo, hi), S.x, S.b)
    Ecf = l2(c1[fld], slice(lo, hi), S.x, S.b)
    print(f"  corner {fld} L2: ground={Egf:.6e} cand={Ecf:.6e} ratio={Ecf / Egf:.4f}")

# ---------------------------------------------------------------- task 4
print("\n=== TASK 4: corner departure of A ===")
WX = pc.CornerRegSolver.WX_REF
pin = WX * np.cos(S.b)
for name, d in (("ground", g), ("candidate", c1)):
    A = d["A"]
    print(f"{name}: |A(0,:)-WXcosb|_inf = {np.max(np.abs(A[0, :] - pin)):.3e}")
    for j in (1, 2, 3):
        diff = A[j, :] - A[0, :]
        print(f"  node {j} (xi={S.x[j]:.6f}): |A(j)-A(0)|_inf={np.max(np.abs(diff)):.6e}"
              f"  rms={np.sqrt(np.mean(diff ** 2)):.6e}"
              f"  rms/xi_j={np.sqrt(np.mean(diff ** 2)) / S.x[j]:.6e}")
# spectral radial derivative at the corner (panel-0 differentiation block)
D0 = S.Dx[offs[0]:offs[1], offs[0]:offs[1]].toarray()
for name, d in (("ground", g), ("candidate", c1)):
    dA = D0 @ d["A"][offs[0]:offs[1], :]
    dA0 = dA[0, :]
    # projection on cos b and residual shape
    cb = np.cos(S.b)
    coef = float(np.dot(dA0, cb) / np.dot(cb, cb))
    resid = float(np.linalg.norm(dA0 - coef * cb) / np.linalg.norm(dA0))
    print(f"{name}: dA/dxi(0,:) rms={np.sqrt(np.mean(dA0 ** 2)):.6e}  "
          f"inf={np.max(np.abs(dA0)):.6e}  cosb-coef={coef:+.6e}  "
          f"non-cosb frac={resid:.3e}")
g_dA0 = (D0 @ g["A"][offs[0]:offs[1], :])[0, :]
c_dA0 = (D0 @ c1["A"][offs[0]:offs[1], :])[0, :]
print(f"ratio rms dA/dxi(0) candidate/ground = "
      f"{np.sqrt(np.mean(c_dA0 ** 2)) / np.sqrt(np.mean(g_dA0 ** 2)):.6f}")

# B corner check for completeness
THXX = pc.CornerRegSolver.THXX_REF
pinB = 0.5 * THXX * np.cos(S.b) ** 2
for name, d in (("ground", g), ("candidate", c1)):
    print(f"{name}: |B(0,:)-THXX/2 cos^2 b|_inf = "
          f"{np.max(np.abs(d['B'][0, :] - pinB)):.3e}")

# max|field| sanity
for name, d in (("ground", g), ("candidate", c1)):
    m = max(np.max(np.abs(d[f])) for f in ("A", "B", "P"))
    print(f"{name}: max|field| = {m:.6f}")
