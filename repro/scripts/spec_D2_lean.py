"""LEAN, STREAMING version of the two decisive D questions.
(1) G1 clause-2: does deflating the grading shadow move ||R(z)||?
(2) RHP: real-axis + imaginary-axis resolvent scan, which bounds where an eigenvalue
    could hide via ||R(z)|| >= 1/dist(z, spectrum).
"""
import importlib.util, pathlib, sys, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
s = importlib.util.spec_from_file_location("pc", str(H / "polar_cornerreg.py"))
pc = importlib.util.module_from_spec(s); sys.modules["pc"] = pc; s.loader.exec_module(pc)
d = np.load(SCR / "hunt_fields/rung_00_a-0.344712.npz")
a, zr = float(d["a"]), d["z"]
S = pc.CornerRegSolver(edges=(0.0,2.0,15.0,25.0), degs=(16,40,12), Nb=36, eps_b=1e-4, alpha=a)
S.adopt_seed(zr)
Nx, Nb = S.Nx, S.Nb; n2 = Nx*Nb
J = S.jacobian(zr).tocsr(); N = J.shape[0]
liveT = np.setdiff1d(np.arange(n2), np.union1d(S.rT_pin, S.rT_c0))
fr = np.concatenate([liveT, n2+liveT]); nf = fr.size
mask = np.zeros(N); mask[fr]=1.0
E = sp.diags(mask, format="csc"); Jc = J.tocsc()
Cg = np.asarray(J[[N-2,N-1],:].todense())[:, fr]
Qc,_ = np.linalg.qr(Cg.T)
w1 = np.load(SCR/"quotient_state.npz")["w1"]; w1 = w1/np.linalg.norm(w1)
print(f"n_f={nf}  h_id={S.h_id(zr):+.4e}  ||F||_rms="
      f"{np.linalg.norm(S.residual(zr))/np.sqrt(N):.3e}", flush=True)

def Rn(zz, extra=None, iters=60, tol=1e-10):
    Q = Qc if extra is None else np.linalg.qr(np.column_stack([Qc, extra]))[0]
    Pk = lambda v: v - Q@(Q.conj().T@v)
    lu = spla.splu((zz*E - Jc).tocsc())
    def R(f,h=False):
        r=np.zeros(N,dtype=complex); r[fr]=f
        return lu.solve(r, trans=("H" if h else "N"))[fr]
    v = Pk(np.random.default_rng(0).standard_normal(nf).astype(complex)); v/=np.linalg.norm(v)
    sold=0.0
    for it in range(iters):
        y=Pk(R(v)); w=Pk(R(y,True)); nn=np.linalg.norm(w); snew=np.sqrt(nn)
        if it>4 and abs(snew-sold)<tol*snew: return snew, it+1
        v=w/nn; sold=snew
    return sold, iters

print("\n(1) DEFLATION TEST  --  does removing the grading shadow move ||R(z)||?", flush=True)
print(f"{'z':>16s} {'structural only':>20s} {'+grading deflated':>20s} {'rel change':>12s}", flush=True)
for zz in (0.0+0j, 0.5+0j, 1.0+0j, 0.0+1.0j, -0.5+0j):
    s0,i0 = Rn(zz); s1,i1 = Rn(zz, extra=w1[:,None])
    print(f"{zz.real:+8.2f}{zz.imag:+6.2f}i {s0:>20.10e} {s1:>20.10e} {abs(s1-s0)/s0:>12.3e}",
          flush=True)

print("\n(2) RHP EXCLUSION SCAN  --  ||R(z)|| >= 1/dist(z,spec) gives an exclusion radius",
      flush=True)
print(f"{'z':>16s} {'||R(z)||':>14s} {'1/||R||  = exclusion radius':>30s}", flush=True)
pts = [0.0+0j, 0.0+0.5j, 0.0+1j, 0.0+2j, 0.0+4j, 0.0+8j, 0.0+16j, 0.0+64j, 0.0+256j,
       0.25+0j, 0.5+0j, 1.0+0j, 2.0+0j, 4.0+0j, 8.0+0j, 16.0+0j, 64.0+0j, 256.0+0j,
       1.0+1.0j, 4.0+4.0j, 16.0+16.0j, 100.0+0j, 400.0+0j, 600.0+0j]
rec=[]
for zz in pts:
    t0=time.time(); s,it = Rn(zz)
    rec.append((zz,s))
    print(f"{zz.real:+8.2f}{zz.imag:+6.2f}i {s:>14.5e} {1.0/s:>30.5e}   [{it} it, {time.time()-t0:.0f}s]",
          flush=True)
np.save(SCR/"spec_D2_scan.npy", np.array([(z.real,z.imag,s) for z,s in rec]))
