import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pc","/Users/epagogellc/parzival/boussinesq/polar_corner.py")
pc=importlib.util.module_from_spec(sp); sys.modules["pc"]=pc; sp.loader.exec_module(pc)
C=pc.Corner(48,48,25.0)
dO,dB,Pt,cl,cw,cond=C.rhs(C.Ot0,C.Bt0)
p=np.abs(dO).max(axis=1)
print("residual by xi node (first 12 and last 4):")
for i in list(range(12))+[C.nx-4,C.nx-3,C.nx-2,C.nx-1]:
    print("  i=%2d xi=%8.4f r=%11.4g g=%9.6f  max|dOt|=%11.4e  |Ot|=%10.4e |Bt|=%10.4e |Pt|=%10.4e"
          % (i,C.x[i],C.r[i],C.g[i],p[i],np.abs(C.Ot0[i]).max(),
             np.abs(C.Bt0[i]).max(),np.abs(C.Pt0[i]).max()))
print("\nEXPECTED corner scalings (measured constants: w_x(0)=1.19620, th_xx(0)=1.79819):")
print("  Om ~ w_x(0) r cos b   => Ot ~ r^(1-a) ~ r^1.342 (a=%.4f)" % C.a0)
print("  B  ~ th_xx(0)/2 r^2   => Bt ~ r^(1-2a) ~ r^1.685")
print("  Psi ~ r^2             => Pt ~ r^(-a)  ~ r^0.342 ... times r^2 = r^1.658")
print("\n  measured log-slopes of the SEED near the corner (d log F / d log r):")
for nm,A,pred in (("Ot",C.Ot0,1-C.a0),("Bt",C.Bt0,1-2*C.a0),("Pt",C.Pt0,2-C.a0)):
    j=C.nb//2
    v=np.abs(A[1:8,j]); rr=C.r[1:8]
    m=v>0
    if m.sum()>3:
        sl=np.polyfit(np.log(rr[m]),np.log(v[m]),1)[0]
        print("     %s: measured %+.4f   predicted %+.4f   %s"
              % (nm,sl,pred,"OK" if abs(sl-pred)<0.25 else "<-- MISMATCH"))
