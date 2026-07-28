import sys,os,numpy as np, scipy.linalg as sla
os.chdir("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
sys.path.insert(0,os.getcwd()); import q345, we_range
real,S,a,z = q345.spectrum.load_production("A")
Nb=S.Nb
print("  node  xi      k        maxRe W(S)     minRe W(S)   maxRe with J_AB=0", flush=True)
for i,k in [(20,1e3),(20,1e4),(20,1e5),(20,1e6),(50,1e5),(68,1e5)]:
    Sk = we_range.symbol(S,z,i,k)
    H=0.5*(Sk+Sk.conj().T); w=sla.eigvalsh(H)
    S2=Sk.copy(); S2[:Nb,Nb:]=0.0            # kill only the A<-B coupling block
    H2=0.5*(S2+S2.conj().T); w2=sla.eigvalsh(H2)
    print(f"  {i:4d} {S.x[i]:6.3f} {k:8.0e}  {w[-1]:+12.5e}  {w[0]:+12.5e}   {w2[-1]:+12.5e}", flush=True)
# slope: maxRe / k  ->  the coupling constant  E1*G1*xi/2 with the cos(beta) profile
for i in (20,50,68):
    k=1e6; Sk=we_range.symbol(S,z,i,k); w=sla.eigvalsh(0.5*(Sk+Sk.conj().T))[-1]
    G1=float(S.G1c[i]); xi=float(S.x[i]); E1=float(np.exp(S.a0*xi)/G1)
    print(f"  node {i:3d} xi={xi:6.3f}:  maxRe/k = {w/k:.6e}   E1*G1*xi/2 = {E1*G1*xi/2:.6e}", flush=True)
