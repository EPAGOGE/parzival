"""Morphology fingerprint: ground root (rung_00) vs candidate branch root
(find_half / branch1_eps1e-4).  Measurement only -- seconds of compute."""
import importlib.util, sys
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad"
HF = SCRATCH + "/hunt_fields"

spec = importlib.util.spec_from_file_location(
    'pc', '/Users/epagogellc/parzival/boussinesq/polar_cornerreg.py')
pc = importlib.util.module_from_spec(spec); sys.modules['pc'] = pc
spec.loader.exec_module(pc)
S = pc.CornerRegSolver(edges=(0.0, 2.0, 15.0, 25.0), degs=(16, 40, 12),
                       Nb=36, eps_b=1e-4, alpha=-0.344712)
x, b = S.x, S.b
Nx, Nb = len(x), len(b)
n2 = Nx * Nb

def load(name):
    d = np.load(f"{HF}/{name}.npz")
    z = d['z']
    A = z[:n2].reshape(Nx, Nb)
    B = z[n2:2*n2].reshape(Nx, Nb)
    P = z[2*n2:3*n2].reshape(Nx, Nb)
    cl, cw = z[3*n2], z[3*n2+1]
    return dict(A=A, B=B, P=P, cl=cl, cw=cw, a=float(d['a']))

G  = load('rung_00_a-0.344712')     # ground root, its own alpha
C  = load('find_half')              # candidate root at frozen a=alpha_1
Cp = load('branch1_eps1e-4')        # secant-polished branch root
G9 = load('rung_09_a-0.414493')     # ground-branch walk near candidate's a
G10 = load('rung_10_a-0.422247')

print(f"grid: Nx={Nx} Nb={Nb}  x[0..3]={x[:4]}  b[0]={b[0]:.6f} b[-1]={b[-1]:.6f}")
print(f"a: ground={G['a']:.6f} cand={C['a']:.7f} polished={Cp['a']:.8f}")
print(f"cl,cw ground = {G['cl']:.8f}, {G['cw']:.8f}")
print(f"cl,cw cand   = {C['cl']:.8f}, {C['cw']:.8f}")
print(f"cl,cw polish = {Cp['cl']:.8f}, {Cp['cw']:.8f}")

# de-duplicate interface nodes for radial line scans (unique xi, keep first)
_, uidx = np.unique(x, return_index=True)
xu = x[uidx]

def radial_profile(F, j):
    return F[uidx, j]

def sign_changes(v, rel_floor=1e-6):
    """count sign changes ignoring values below rel_floor*max|v|"""
    thr = rel_floor * np.max(np.abs(v))
    w = v[np.abs(v) > thr]
    if len(w) < 2:
        return 0
    return int(np.sum(np.sign(w[:-1]) != np.sign(w[1:])))

def n_extrema(v, rel_prom=1e-3):
    """count interior local extrema with prominence > rel_prom*max|v|"""
    thr = rel_prom * np.max(np.abs(v))
    d = np.diff(v)
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0] + 1
    # prominence proxy: |v[i] - nearest neighbor value| beyond threshold
    keep = [i for i in idx
            if min(abs(v[i]-v[i-1]), abs(v[i]-v[i+1])) > 0 and
               abs(v[i]) > thr]
    return len(keep)

# ---- 1. radial structure at several beta stations -------------------------
print("\n== 1. RADIAL STRUCTURE (sign changes / interior extrema along xi) ==")
# beta stations: near-wall j=1, quarter, mid-wedge, three-quarter, near-axis
stations = [1, Nb//4, Nb//2, 3*Nb//4, Nb-2]
hdr = "field root   " + "".join([f"  b={b[j]:.3f}(j={j})" for j in stations])
print(hdr)
for fld in ('A', 'B'):
    for nm, R in (('ground', G), ('cand', C), ('polish', Cp),
                  ('grnd@a1(r09)', G9)):
        sc = [sign_changes(radial_profile(R[fld], j)) for j in stations]
        ex = [n_extrema(radial_profile(R[fld], j)) for j in stations]
        print(f"{fld}  {nm:13s} sc={sc}  extrema={ex}")

# ---- 2. localize the distance per panel per field -------------------------
print("\n== 2. PANEL-LOCALIZED RELATIVE L2 DIFFERENCE ==")
panels = [("corner xi<2", x <= 2.0), ("mid 2-15", (x > 2.0) & (x <= 15.0)),
          ("outer 15-25", x > 15.0)]

def panel_report(Ra, Rb, tag):
    print(f"-- {tag}")
    dz_tot = 0.0; nz_tot = 0.0
    rows = []
    for fld in ('A', 'B', 'P'):
        for pnm, mask in panels:
            d = Ra[fld][mask] - Rb[fld][mask]
            g = Rb[fld][mask]
            dz = float(np.sum(d**2)); nz = float(np.sum(g**2))
            rows.append((fld, pnm, dz, nz))
            dz_tot += dz; nz_tot += nz
    for fld, pnm, dz, nz in rows:
        print(f"   {fld} {pnm:12s} relL2(panel)={np.sqrt(dz/nz):9.4f}"
              f"   share of total diff^2 = {dz/dz_tot:8.5f}")
    print(f"   GLOBAL field relative distance = {np.sqrt(dz_tot/nz_tot):.6f}")
    # include cl,cw in the z-style distance
    dcl = (Ra['cl']-Rb['cl'])**2 + (Ra['cw']-Rb['cw'])**2
    ncl = Rb['cl']**2 + Rb['cw']**2
    print(f"   z-style distance incl (cl,cw)  = "
          f"{np.sqrt((dz_tot+dcl)/(nz_tot+ncl)):.6f}")

panel_report(C, G, "candidate(find_half) vs GROUND(rung_00)")
panel_report(C, G9, "candidate(find_half) vs ground-branch @a=-0.4145 (rung_09)")
# interpolate ground branch to a=-0.4168236 between rung_09 and rung_10
t = (C['a'] - G9['a']) / (G10['a'] - G9['a'])
Gi = {f: (1-t)*G9[f] + t*G10[f] for f in ('A','B','P')}
Gi['cl'] = (1-t)*G9['cl'] + t*G10['cl']; Gi['cw'] = (1-t)*G9['cw'] + t*G10['cw']
panel_report(C, Gi, f"candidate vs ground-branch INTERPOLATED to a={C['a']:.7f} (t={t:.3f})")

# ---- 3. amplitude anatomy -------------------------------------------------
print("\n== 3. AMPLITUDE ANATOMY ==")
for nm, R in (('ground', G), ('cand', C), ('polish', Cp), ('grnd@r09', G9)):
    print(f"{nm:9s} max|A|={np.max(np.abs(R['A'])):.6f} "
          f"max|B|={np.max(np.abs(R['B'])):.6f} "
          f"max|P|={np.max(np.abs(R['P'])):.6f} "
          f"cl={R['cl']:.6f} cw={R['cw']:.6f} cw/cl={R['cw']/R['cl']:.6f}")
# location of max|field|
for nm, R in (('ground', G), ('cand', C)):
    for fld in ('A','B','P'):
        i, j = np.unravel_index(np.argmax(np.abs(R[fld])), (Nx, Nb))
        print(f"  {nm} argmax|{fld}| at xi={x[i]:.4f}, b={b[j]:.4f}, val={R[fld][i,j]:+.6f}")

print("\n-- sign structure of A --")
for nm, R in (('ground', G), ('cand', C), ('polish', Cp)):
    A = R[fld] if False else R['A']
    neg = A < -1e-6*np.max(np.abs(A))
    frac = neg.mean()
    print(f"{nm:9s} frac gridpoints A<0 (rel thr 1e-6): {frac:.5f}  "
          f"min A = {A.min():+.6e}  max A = {A.max():+.6e}")
    if neg.any():
        ii, jj = np.where(neg)
        print(f"          A<0 region: xi in [{x[ii].min():.3f},{x[ii].max():.3f}], "
              f"b in [{b[jj].min():.4f},{b[jj].max():.4f}], "
              f"deepest A={A[ii,jj].min():+.4e} at xi={x[ii[np.argmin(A[ii,jj])]]:.3f}")
# same for B and P sign structure
for fldn in ('B','P'):
    for nm, R in (('ground', G), ('cand', C)):
        F = R[fldn]
        neg = (F < -1e-6*np.max(np.abs(F))).mean()
        pos = (F > 1e-6*np.max(np.abs(F))).mean()
        print(f"{fldn} {nm:9s} frac<0={neg:.5f} frac>0={pos:.5f} "
              f"min={F.min():+.4e} max={F.max():+.4e}")

# ---- 4. beta-harmonic content --------------------------------------------
print("\n== 4. BETA HARMONICS (least-squares projection at fixed xi) ==")
# A: odd cosines cos((2k+1)b); B: even cosines cos(2kb) incl k=0; P: sin(2kb)
def ls_coeffs(v, basis_fns):
    M = np.stack([f(b) for f in basis_fns], axis=1)
    c, *_ = np.linalg.lstsq(M, v, rcond=None)
    resid = v - M @ c
    return c, np.linalg.norm(resid)/max(np.linalg.norm(v), 1e-300)

A_basis = [lambda bb, m=m: np.cos((2*m+1)*bb) for m in range(5)]
B_basis = [lambda bb, m=m: np.cos(2*m*bb) for m in range(5)]
P_basis = [lambda bb, m=m: np.sin(2*m*bb) for m in range(1, 6)]
A_names = [f"cos{2*m+1}b" for m in range(5)]
B_names = [f"cos{2*m}b" for m in range(5)]
P_names = [f"sin{2*m}b" for m in range(1, 6)]

xi_stations = [0.5, 1.0, 2.0, 4.0, 8.0]
for fld, basis, names in (('A', A_basis, A_names), ('B', B_basis, B_names),
                          ('P', P_basis, P_names)):
    print(f"-- {fld} onto {names}")
    for xs in xi_stations:
        i = np.argmin(np.abs(x - xs))
        line = f"  xi={x[i]:5.2f}: "
        for nm, R in (('ground', G), ('cand', C)):
            c, r = ls_coeffs(R[fld][i, :], basis)
            cn = c / (np.max(np.abs(c)) if np.max(np.abs(c)) > 0 else 1)
            top = np.abs(c) / max(np.abs(c).sum(), 1e-300)
            line += (f"{nm}: c={np.array2string(c, precision=4, suppress_small=True)} "
                     f"resid={r:.2e}  | ")
        print(line)

# harmonic ENERGY ratio: higher-harmonic fraction |c_{k>=1}|/|c_0| for A
print("\n-- A higher-harmonic fraction sum|c_(3b,5b,..)| / |c_(cos b)| --")
for xs in xi_stations:
    i = np.argmin(np.abs(x - xs))
    out = f"  xi={x[i]:5.2f}: "
    for nm, R in (('ground', G), ('cand', C), ('polish', Cp)):
        c, r = ls_coeffs(R[fld][i, :], A_basis) if False else ls_coeffs(R['A'][i, :], A_basis)
        out += f"{nm}={np.sum(np.abs(c[1:]))/abs(c[0]):.4f}  "
    print(out)
