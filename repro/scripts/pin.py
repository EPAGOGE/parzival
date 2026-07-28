import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py"); pm=M("pm","polar_march.py")
C=pc.Corner(64,64,25.0); Mp=pm.March(64,64,-2.0,25.0,filter_on=False)
# recompute the pieces by hand in both frames
def pieces_corner(C):
    Pt=C.poisson(C.Ot0); Ot,Bt=C.Ot0,C.Bt0
    Ox,Ob=C.dx(Ot),C.db(Ot); Bx,Bb=C.dx(Bt),C.db(Bt); Px,Pb=C.dx(Pt),C.db(Pt)
    a,mu,G=C.a0,C.mu,C.G
    adv=(Px+mu*Pt)*Ob-Pb*(Ox+a*Ot)
    src=G*C.cosb*(Bx+(1+2*a)*Bt)-C.sinb*Bb
    return Pt,adv,src
def pieces_log(Mp):
    Pt=Mp.poisson.solve(Mp.Ot0); Ot,Bt=Mp.Ot0,Mp.Bt0
    Os,Ob=Mp.ds(Ot),Mp.db(Ot); Bs,Bb=Mp.ds(Bt),Mp.db(Bt); Ps,Pb=Mp.ds(Pt),Mp.db(Pt)
    a,mu=Mp.a0,Mp.mu
    adv=(Ps+mu*Pt)*Ob-Pb*(Os+a*Ot)
    src=Mp.cosb*(Bs+(1+2*a)*Bt)-Mp.sinb*Bb
    return Pt,adv,src
Pc,advc,srcc=pieces_corner(C); Pl,advl,srcl=pieces_log(Mp)
print("matched physical r, mid-beta.  xi-frame vs log-polar:\n")
print("  %8s %5s | %12s %12s | %12s %12s | %12s %12s"
      %("r","g","Pt(xi)","Pt(log)","adv(xi)","adv(log)","src(xi)","src(log)"))
for tgt in (1e2,1e4,1e6,1e8,1e10):
    i=int(np.argmin(np.abs(C.r-tgt))); j=int(np.argmin(np.abs(np.exp(Mp.s)-tgt))); k=C.nb//2
    print("  %8.0e %5.3f | %12.5e %12.5e | %12.5e %12.5e | %12.5e %12.5e"
          %(tgt,C.g[i],Pc[i,k],Pl[j,k],advc[i,k],advl[j,k],srcc[i,k],srcl[j,k]))
print("\n  Ot0 and Bt0 at matched r (should agree; ratio g^a and g^(1+2a) -> 1):")
for tgt in (1e2,1e6,1e10):
    i=int(np.argmin(np.abs(C.r-tgt))); j=int(np.argmin(np.abs(np.exp(Mp.s)-tgt))); k=C.nb//2
    print("   r=%.0e  Ot: %.6e vs %.6e   Bt: %.6e vs %.6e"
          %(tgt,C.Ot0[i,k],Mp.Ot0[j,k],C.Bt0[i,k],Mp.Bt0[j,k]))
