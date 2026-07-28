import sys, numpy as np, logging
for n in list(logging.root.manager.loggerDict): logging.getLogger(n).setLevel(logging.ERROR)
sys.path.insert(0,'/Users/epagogellc/parzival/boussinesq')
import polar_radial_gate as G
alpha,gk,beta,g1,M = G.alpha_and_gk(); mu=2.0+alpha
print(f"alpha={alpha:+.8f}  s in [{G.S0},{G.S1}]  NS={G.NS}  span=e^(mu*ds)={np.exp(mu*(G.S1-G.S0)):.3e}")
print(f"{'k':>3s} {'RAW globalL2':>13s} {'RAW per-pt':>12s} | {'SUB globalL2':>13s} {'SUB per-pt':>12s}")
rw,rp,sw,sp=[],[],[],[]
for i,k in enumerate(range(1,G.KMAX+1)):
    ck=gk[i]/((2.0*k)**2-mu**2)
    sg,A,_=G.solve_raw(k,gk[i],alpha);  eA=ck*np.exp(mu*sg)
    sg2,P,_=G.solve_subst(k,gk[i],alpha); eP=ck*np.ones_like(sg2)
    gR=np.linalg.norm(A-eA)/np.linalg.norm(eA)           # the gate's metric
    pR=np.max(np.abs(A-eA)/np.abs(eA))                   # per-point relative
    gS=np.linalg.norm(P-eP)/np.linalg.norm(eP)
    pS=np.max(np.abs(P-eP)/np.abs(eP))
    rw.append(gR);rp.append(pR);sw.append(gS);sp.append(pS)
    print(f"{k:3d} {gR:13.3e} {pR:12.3e} | {gS:13.3e} {pS:12.3e}")
print(f"\nworst RAW  : global L2 = {max(rw):.3e}   PER-POINT = {max(rp):.3e}")
print(f"worst SUBST: global L2 = {max(sw):.3e}   PER-POINT = {max(sp):.3e}")
print(f"\n=> the gate reported RAW {max(rw):.1e} and concluded 'RAW is not disqualified'.")
print(f"   Per-point, RAW is {max(rp)/max(sp):.1e}x worse than SUBST.")
