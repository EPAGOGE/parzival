import numpy as np, logging, sys
logging.getLogger('dedalus').setLevel(logging.ERROR)
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
import polar_tau2d_gate as G

print("SUBST two-mode error + max|tau| vs resolution (is the growth real?)")
print(f"{'Ns':>5s} {'Nb':>5s} {'per-s rel':>12s} {'max|tau|':>11s}")
for Ns,Nb in [(32,24),(48,32),(64,48),(96,64),(128,96),(160,96),(192,128),(256,128)]:
    num,ex,tm,_ = G.solve_manufactured(Ns,Nb,[(1,1.0),(2,0.3)],subst=True)
    g,p = G.errors(num,ex)
    print(f"{Ns:5d} {Nb:5d} {p:12.3e} {tm:11.3e}")

print("\nfix Nb=32, vary Ns  (isolate the s direction)")
for Ns in [32,48,64,96,128,192,256]:
    num,ex,tm,_ = G.solve_manufactured(Ns,32,[(1,1.0),(2,0.3)],subst=True)
    g,p = G.errors(num,ex)
    print(f"  Ns={Ns:4d} per-s={p:.3e} maxtau={tm:.3e}")
print("\nfix Ns=64, vary Nb  (isolate the beta direction)")
for Nb in [16,24,32,48,64,96,128]:
    num,ex,tm,_ = G.solve_manufactured(64,Nb,[(1,1.0),(2,0.3)],subst=True)
    g,p = G.errors(num,ex)
    print(f"  Nb={Nb:4d} per-s={p:.3e} maxtau={tm:.3e}")

print("\ncond(L) growth, (2,2) construction")
for Ns,Nb in [(24,16),(32,24),(48,32),(64,32)]:
    a = G.rank_anatomy(Ns,Nb,2,2)
    print(f"  {Ns}x{Nb} dim={a['dim']:5d} nullity={a['nullity']} cond={a['cond']:.3e}")
