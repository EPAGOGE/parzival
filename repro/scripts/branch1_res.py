"""Resolution study on the candidate unstable root: does alpha march toward
alpha_1 = -0.4168236 as the discretization deepens?  Warm-start from the saved
eps=1e-5 branch field; secant-polish h(a) at each config.  The attribution gate:
sub-percent gap WITH provenance (monotone resolution trend)."""
import importlib.util, pathlib, sys, time
import numpy as np
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
def modp(n,p):
    sp_=importlib.util.spec_from_file_location(n,str(p)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
bh = modp("bh", SCR/"branch_hunt.py"); pc = bh.pc
A1 = -0.4168236
d = np.load(SCR/"hunt_fields/branch1_eps1e-05.npz")
a_prev, z_prev = float(d["a"]), d["z"]
base = dict(edges=(0.0,2.0,15.0,25.0), Nb=36, eps_b=1e-5)
print(f"{'degs':>12} {'branch alpha':>13} {'vs a1':>8} {'||F||':>9} {'secs':>5}", flush=True)
for degs in ((16,40,12),(20,48,12),(24,56,12),(28,56,12)):
    t0=time.time()
    S = pc.CornerRegSolver(**base, degs=degs, alpha=a_prev)
    # interpolate warm field: same Nb; radial grids differ -> rebuild via newton from
    # nearest field if sizes match, else from scratch secant seeded at a_prev
    def solve_at(a, z0):
        S.set_alpha(a)
        z,f,r,taken = S.newton(z0=z0)
        if taken==0 or r>1e-9: return None,None,r
        return float(z[-1])/float(z[-2]), z, r
    n_expected = 3*S.Nx*S.Nb+2
    z0 = z_prev.copy() if z_prev.size==n_expected else None
    if z0 is None:
        # re-hunt at this resolution: anchor + deflate + half start
        S.set_alpha(A1)
        za,fa,ra,ta = S.newton(z0=None)
        zh = za.copy(); zh[:-2] *= 0.5
        z,r,taken,ok = bh.deflated_newton(S, zh, [za], steps=80)
        if not ok:
            print(f"{str(degs):>12}  RE-HUNT DRY ||F||={r:.2e}", flush=True); continue
        z0 = z
    a0, z = a_prev, z0
    c, z2, r = solve_at(a0, z)
    if c is None: print(f"{str(degs):>12}  FAILED ||F||={r:.1e}", flush=True); continue
    h0 = c - a0; z = z2; a1v = a0 + h0
    for it in range(12):
        c, z2, r = solve_at(a1v, z)
        if c is None: break
        h1 = c - a1v; z = z2
        if abs(h1) < 1e-9: break
        a0,h0,a1v = a1v,h1, a1v - h1*(a1v-a0)/(h1-h0)
    print(f"{str(degs):>12} {a1v:+13.8f} {100*(a1v-A1)/abs(A1):+7.3f}% {r:9.2e} {time.time()-t0:5.0f}", flush=True)
    a_prev, z_prev = a1v, z
    np.savez(SCR/f"hunt_fields/branch1_deg{degs[0]}_{degs[1]}.npz", z=z, a=a1v)
print("done", flush=True)
