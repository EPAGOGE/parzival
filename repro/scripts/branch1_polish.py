"""Polish the candidate unstable root: secant on h(a) along ITS branch (warm from the
found field), re-measuring dh/da here rather than transferring the n=0 invariance
(the I2 rule). Then the eps ladder 1e-4 -> 1e-5 and a deg0=20 rung for the (N,eps) study.
Labels: alpha_1 = -0.4168236 (DeepMind), CHL stage-2 = -0.40834."""
import importlib.util, pathlib, sys, time
import numpy as np
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
def modp(n,p):
    sp_=importlib.util.spec_from_file_location(n,str(p)); m=importlib.util.module_from_spec(sp_)
    sys.modules[n]=m; sp_.loader.exec_module(m); return m
bh = modp("bh", SCR/"branch_hunt.py"); pc = bh.pc
A1 = -0.4168236
zf = np.load(SCR/"hunt_fields/find_half.npz")["z"]

def solve_at(S, a, z0):
    S.set_alpha(a)
    z,f,r,taken = S.newton(z0=z0.copy())
    if taken==0 or r>1e-9: return None,None,r
    return float(z[-1])/float(z[-2]), z, r

print("SECANT on h(a) along the candidate branch, eps=1e-4, (16,40,12)/Nb36:", flush=True)
S = pc.CornerRegSolver(**bh.CFG, alpha=A1)
a0, z = A1, zf
h0, z, r = solve_at(S, a0, z)
h0 = h0 - a0
print(f"  a={a0:+.7f} h={h0:+.5f} ||F||={r:.1e}", flush=True)
a1 = a0 + h0          # first step: fixed-point suggestion
for it in range(12):
    cw_cl, z2, r = solve_at(S, a1, z)
    if cw_cl is None: print(f"  a={a1:+.7f} FAILED ||F||={r:.1e}", flush=True); break
    h1 = cw_cl - a1; z = z2
    print(f"  a={a1:+.7f} h={h1:+.6f} ||F||={r:.1e}  dh/da so far: "
          f"{(h1-h0)/(a1-a0):+.3f}", flush=True)
    if abs(h1) < 1e-9: break
    a0,h0,a1 = a1,h1, a1 - h1*(a1-a0)/(h1-h0)
astar = a1
print(f"\n  BRANCH ALPHA at eps=1e-4: {astar:+.8f}  (alpha_1={A1}, gap {astar-A1:+.2e})", flush=True)
np.savez(SCR/"hunt_fields/branch1_eps1e-4.npz", z=z, a=astar)

print("\nEPS LADDER on the branch (warm continuation):", flush=True)
prev_a, prev_z = astar, z
rows=[(1e-4,astar)]
for eps in (5e-5, 2.5e-5, 1e-5):
    S2 = pc.CornerRegSolver(edges=bh.CFG["edges"], degs=bh.CFG["degs"],
                            Nb=bh.CFG["Nb"], eps_b=eps, alpha=prev_a)
    a0z, zz = prev_a, prev_z
    hh, zz, r = solve_at(S2, a0z, zz)
    if hh is None: print(f"  eps={eps:.0e} FAILED", flush=True); break
    hh -= a0z; a1z = a0z + hh
    for it in range(12):
        c, z2, r = solve_at(S2, a1z, zz)
        if c is None: break
        h1 = c - a1z; zz = z2
        if abs(h1) < 1e-9: break
        a0z,hh,a1z = a1z,h1, a1z - h1*(a1z-a0z)/(h1-hh)
    print(f"  eps={eps:.0e}: branch alpha = {a1z:+.8f}  ||F||={r:.1e}", flush=True)
    rows.append((eps,a1z)); prev_a, prev_z = a1z, zz
    np.savez(SCR/f"hunt_fields/branch1_eps{eps:.0e}.npz", z=zz, a=a1z)
if len(rows)>=3:
    es=np.array([p[0] for p in rows]); al=np.array([p[1] for p in rows])
    lin=float(np.polyfit(es[-3:],al[-3:],1)[-1])
    print(f"\n  eps->0 (linear, last 3): {lin:+.8f}   alpha_1={A1}   gap {lin-A1:+.2e}", flush=True)
print("done", flush=True)
