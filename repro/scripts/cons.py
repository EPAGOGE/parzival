import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py")
print("CONSERVING GAUGE vs L2 GAUGE, corner frame, 48x48, filter on, dt 5e-4\n")
for mode in ("l2","conserve"):
    C=pc.Corner(48,48,25.0,filter_on=True); C.gauge_mode=mode
    I=(slice(2,-2),slice(2,-2))
    r0=np.abs(C.rhs(C.Ot0,C.Bt0)[0][I]).max()
    wx0=float((C.Dx@C.Ot0)[0,0]); th0=float((C.Dx2@C.Bt0)[0,0])
    print("  --- %s ---   res0=%.4e  w_x(0)=%.6f  th_xx(0)=%.6f"%(mode,r0,wx0,th0))
    print("  %7s %7s %11s %10s %10s %11s %11s"%("step","tau","max|dOt|","c_l","alpha","w_x drift","th_xx drift"))
    bad=False
    for k in range(8001):
        if k%1000==0:
            dO,_,_,cl,cw,_=C.rhs(C.Ot,C.Bt)
            rr=np.abs(dO[I]).max()
            wx=float((C.Dx@C.Ot)[0,0]); th=float((C.Dx2@C.Bt)[0,0])
            print("  %7d %7.2f %11.4e %10.6f %10.6f %11.3e %11.3e"
                  %(k,k*5e-4,rr,cl,cw/cl,abs(wx-wx0)/abs(wx0),abs(th-th0)/abs(th0)),flush=True)
            if not np.isfinite(rr) or rr>1e4: print("   DIVERGED"); bad=True
        if bad: break
        C.step(5e-4)
    print()
