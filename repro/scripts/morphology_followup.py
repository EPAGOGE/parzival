"""Follow-up: extremum locations, branch-vs-a controls on beta harmonics,
and node structure of the DIFFERENCE field (cand - same-a ground branch)."""
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
    return dict(A=z[:n2].reshape(Nx, Nb), B=z[n2:2*n2].reshape(Nx, Nb),
                P=z[2*n2:3*n2].reshape(Nx, Nb), cl=z[3*n2], cw=z[3*n2+1],
                a=float(d['a']))

G = load('rung_00_a-0.344712'); C = load('find_half')
Cp = load('branch1_eps1e-4')
G9 = load('rung_09_a-0.414493'); G10 = load('rung_10_a-0.422247')
t = (C['a'] - G9['a']) / (G10['a'] - G9['a'])
Gi = {f: (1-t)*G9[f] + t*G10[f] for f in ('A', 'B', 'P')}

_, uidx = np.unique(x, return_index=True)
xu = x[uidx]

def extrema_list(v, rel_prom=1e-3):
    thr = rel_prom * np.max(np.abs(v))
    d = np.diff(v)
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0] + 1
    return [(xu[i], v[i], 'max' if v[i] > v[i-1] else 'min')
            for i in idx if abs(v[i]) > thr]

print("== EXTREMUM LOCATIONS along xi ==")
for fld in ('A', 'B'):
    for j, jnm in ((1, 'near-wall'), (Nb//2, 'mid-wedge')):
        print(f"-- {fld}({jnm} b={b[j]:.3f})")
        for nm, R in (('ground(rung00)', G), ('grndB@a1(interp)', Gi),
                      ('cand(find_half)', C), ('polish', Cp)):
            v = R[fld][uidx, j]
            ex = extrema_list(v)
            s = ", ".join([f"{k}@xi={xi_:.3f} val={val:+.4f}" for xi_, val, k in ex])
            print(f"   {nm:18s} {s}")

print("\n== cos3b COEFFICIENT OF A vs xi: branch control ==")
A_basis = np.stack([np.cos((2*m+1)*b) for m in range(5)], axis=1)
def c_of(R, i):
    c, *_ = np.linalg.lstsq(A_basis, R['A'][i, :], rcond=None)
    return c
print("   xi     ground(c3/c1)  grndB@a1(c3/c1)  cand(c3/c1)   polish(c3/c1)")
for xs in [0.29, 0.44, 0.63, 1.0, 1.38, 2.0, 3.0]:
    i = int(np.argmin(np.abs(x - xs)))
    row = f"  {x[i]:5.2f}  "
    for R in (G, Gi, C, Cp):
        c = c_of(R, i)
        row += f"   {c[1]/c[0]:+9.4f}   "
    print(row)

print("\n== A higher-harmonic fraction: branch control (incl grndB@a1) ==")
for xs in [0.44, 1.0, 2.0, 3.9, 8.0]:
    i = int(np.argmin(np.abs(x - xs)))
    row = f"  xi={x[i]:5.2f}: "
    for nm, R in (('ground', G), ('grndB@a1', Gi), ('cand', C)):
        c = c_of(R, i)
        row += f"{nm}={np.sum(np.abs(c[1:]))/abs(c[0]):.4f}  "
    print(row)

print("\n== DIFFERENCE FIELD d = cand - grndB@a1 (the branch-separation direction) ==")
for fld in ('A', 'B', 'P'):
    D = C[fld] - Gi[fld]
    rel = np.linalg.norm(D) / np.linalg.norm(Gi[fld])
    i, j = np.unravel_index(np.argmax(np.abs(D)), (Nx, Nb))
    print(f" {fld}: relL2={rel:.4f}  max|d|={np.abs(D).max():.4f} at xi={x[i]:.3f}, b={b[j]:.3f}")
    for jj, jnm in ((1, 'near-wall'), (Nb//2, 'mid-wedge'), (Nb-2, 'near-axis')):
        v = D[uidx, jj]
        thr = 1e-4 * np.max(np.abs(v))
        w = v[np.abs(v) > thr]
        sc = int(np.sum(np.sign(w[:-1]) != np.sign(w[1:]))) if len(w) > 1 else 0
        # first crossing location
        cross = None
        for k in range(1, len(v)):
            if abs(v[k]) > thr and abs(v[k-1]) > thr and np.sign(v[k]) != np.sign(v[k-1]):
                cross = xu[k]; break
        print(f"    d{fld}({jnm}): sign changes along xi = {sc}"
              + (f", first crossing xi={cross:.3f}" if cross else "")
              + f", d at xi=0: {v[0]:+.2e}, peak {v[np.argmax(np.abs(v))]:+.4f}@xi={xu[np.argmax(np.abs(v))]:.3f}")

print("\n== RADIAL DECAY: log-slope of near-wall A on outer panel (xi 15-25) ==")
for nm, R in (('ground', G), ('grndB@a1', Gi), ('cand', C)):
    v = R['A'][uidx, 1]
    m = (xu >= 15.0) & (xu <= 25.0) & (v > 0)
    sl = np.polyfit(xu[m], np.log(v[m]), 1)[0]
    print(f"  {nm:10s} d(ln A)/d(xi) on [15,25] near-wall = {sl:+.5f}")

print("\n== corner algebra c = (cl - 2 cw)/4 vs measured P(0,b) sin2b coeff ==")
sin2 = np.sin(2*b)
for nm, R in (('ground', G), ('cand', C), ('polish', Cp)):
    c_pred = (R['cl'] - 2*R['cw'])/4
    c_meas = float(np.dot(R['P'][0, :], sin2) / np.dot(sin2, sin2))
    resid = np.linalg.norm(R['P'][0, :] - c_meas*sin2)/np.linalg.norm(R['P'][0, :])
    print(f"  {nm:8s} c_pred={c_pred:.6f}  c_meas(sin2b fit)={c_meas:.6f} "
          f" ratio={c_meas/c_pred:.6f}  fit resid={resid:.2e}")
