import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py")
print("CHEN-HOU'S ACTUAL POINT GAUGE, now computable because the corner is IN the domain.")
print("  F_pertb_2lev.m:135-137   cl = 4 vx1(1,1)/wx1(1,1) = 2 th_xx(0)/w_x(0)")
print("                           cw = u1dx1(1,1) + cl/2   = u_x(0) + cl/2")
print("  in xi variables at the corner (xi ~ r, e^(p xi) -> 1):")
print("     w_x(0)   = Ot_xi(0, b=0)")
print("     th_xx(0) = Bt_xixi(0, b=0)")
print("     u_x(0)   = -d_b[ Pt_xixi(0,b)/2 ] at b=0     (Psi ~ r^2 h(b), u_x(0) = -h'(0))\n")
for N in (48,64,96):
    C=pc.Corner(N,N,25.0)
    Pt=C.poisson(C.Ot0)
    Ox  = C.Dx @ C.Ot0
    Bxx = C.Dx2 @ C.Bt0
    Pxx = C.Dx2 @ Pt
    wx  = Ox[0,0]
    thxx= Bxx[0,0]
    h   = 0.5*Pxx[0,:]
    ux  = -(C.Db @ h)[0]
    cl  = 2.0*thxx/wx
    cw  = ux + cl/2.0
    print("  N=%3d: w_x(0)=%.6f  th_xx(0)=%.6f  u_x(0)=%.6f" % (N,wx,thxx,ux))
    print("         c_l=%+9.6f (ref %+9.6f, err %.3e)   c_w=%+9.6f (ref %+9.6f, err %.3e)   alpha=%+9.6f"
          %(cl,C.P["cl"],abs(cl-C.P["cl"])/abs(C.P["cl"]),
            cw,C.P["cw"],abs(cw-C.P["cw"])/abs(C.P["cw"]),cw/cl), flush=True)
print("\n  reference corner constants measured in polar_gauge_gate.py:")
print("     w_x(0)=1.19620314  th_xx(0)=1.79819132  u_x(0)=-2.53267418")
