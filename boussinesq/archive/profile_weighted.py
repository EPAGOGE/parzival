"""
Weighted-variable Newton solve for the 2D Boussinesq blowup profile.

WHY WEIGHTED VARIABLES
----------------------
The profile's far field is ALGEBRAIC and two of the three fields GROW:
    Om ~ r^alpha            alpha = c_w/c_l ~ -0.3424   (decays)
    B  ~ r^(1+2alpha)       = r^(+0.3152)               (GROWS)
    Psi~ r^(2+alpha)        = r^(+1.6576)               (GROWS)
`Psi=0` at the outer edge is flatly incompatible (fixed earlier with a homogeneity
condition). But B's growth is worse: B's outer edge is an OUTFLOW characteristic, so no
pointwise condition may legitimately be imposed there -- and left free, ||B|| ran away
(2.2 -> 474) and killed the Jacobian.
FIX: carry the growth ANALYTICALLY. Substitute
    Om = Wo*Ot,  B = Wb*Bt,  Psi = Wp*Pt,
    Wo=(1+r^2)^(alpha/2), Wb=(1+r^2)^((1+2alpha)/2), Wp=(1+r^2)^((2+alpha)/2)
so Ot,Bt,Pt are BOUNDED and tend to angular functions. Growth lives in the prefactor;
the unknowns are well behaved exactly where they previously blew up.

GAUGE: TWO weighted integral gauges, NO pins. There are TWO scaling symmetries and both
must be broken. Pinning c_l does NOT break one -- c_l is a CONSEQUENCE of the profile,
not a normalization -- so pinning it leaves a NEUTRAL direction, hence the near-singular
Jacobian and the sign-flipping c_w seen in every pinned run. The two-gauge configuration
was always the well-conditioned one (residuals fell 10x monotonically); its only problem
was converging to the c_l->0 root, which is a BASIN problem the weights address.
"""
import argparse, json, pathlib
import numpy as np
import dedalus.public as d3

C_L_T, C_W_T = 3.00649898, -1.02942516
GAMMA_T = 2.9205600
ALPHA = C_W_T / C_L_T

def main(N=32, Ybox=8.0, iters=20, damping=0.4, seed_sign=-1.0,
         out="../runs/profile_weighted.json"):
    co = d3.CartesianCoordinates("y1", "y2")
    dist = d3.Distributor(co, dtype=np.float64)
    b1 = d3.ChebyshevT(co["y1"], size=N, bounds=(0, Ybox), dealias=3/2)
    b2 = d3.ChebyshevT(co["y2"], size=N, bounds=(0, Ybox), dealias=3/2)
    y1 = dist.local_grid(b1); y2 = dist.local_grid(b2)
    ey1, ey2 = co.unit_vector_fields(dist)
    Ot = dist.Field(name="Ot", bases=(b1,b2)); Bt = dist.Field(name="Bt", bases=(b1,b2))
    Pt = dist.Field(name="Pt", bases=(b1,b2))
    c_l = dist.Field(name="c_l"); c_w = dist.Field(name="c_w")
    t1=dist.Field(name="t1",bases=b2); t2=dist.Field(name="t2",bases=b2)
    t3=dist.Field(name="t3",bases=b1); t4=dist.Field(name="t4",bases=b1)
    s1=dist.Field(name="s1",bases=b2); q1=dist.Field(name="q1",bases=b2)
    lift1=lambda F,n: d3.Lift(F,b1.derivative_basis(2),n)
    lift2=lambda F,n: d3.Lift(F,b2.derivative_basis(2),n)
    L1=lambda F,n: d3.Lift(F,b1.derivative_basis(1),n)
    d1=lambda F: d3.Differentiate(F,co["y1"]); d2=lambda F: d3.Differentiate(F,co["y2"])
    Y1,Y2=np.meshgrid(np.ravel(y1),np.ravel(y2),indexing="ij"); R2=Y1**2+Y2**2
    def wfield(p):
        f=dist.Field(bases=(b1,b2)); f["g"]=((1+R2)**p).reshape(f["g"].shape); return f
    Wo=wfield(ALPHA/2); Wb=wfield((1+2*ALPHA)/2); Wp=wfield((2+ALPHA)/2)
    y1f=dist.Field(bases=b1); y1f["g"]=y1; y2f=dist.Field(bases=b2); y2f["g"]=y2
    r2f=dist.Field(bases=(b1,b2)); r2f["g"]=R2.reshape(r2f["g"].shape)
    wt=dist.Field(bases=(b1,b2)); wt["g"]=((1+R2)**-2.0).reshape(wt["g"].shape)
    # physical fields as expressions in the tilde unknowns
    Om = Wo*Ot; B = Wb*Bt; Psi = Wp*Pt
    U = d3.skew(d3.grad(Psi))
    # seed: bounded tilde fields with the corner structure (B ~ y1^2, Om odd)
    A0=1.0; C0=C_L_T*A0/4.0
    Bt["g"]=(seed_sign*C0*Y1**2*(1+R2)**-1.0).reshape(Bt["g"].shape)
    Ot["g"]=(seed_sign*A0*Y1*(1+R2)**-0.5).reshape(Ot["g"].shape)
    c_l["g"]=C_L_T; c_w["g"]=C_W_T
    integ=lambda ex: d3.Integrate(d3.Integrate(ex,co["y1"]),co["y2"])
    sval=lambda ex: float(np.ravel(ex.evaluate()["g"])[0])
    E1=sval(integ(wt*Om**2)); E2=sval(integ(wt*r2f*Om**2))
    print(f"  gauges from seed: E1={E1:.6e} E2={E2:.6e}")
    RES_OM = c_w*Om + d1(B) - (c_l*y1f+U@ey1)*d1(Om) - (c_l*y2f+U@ey2)*d2(Om)
    RES_B  = (c_l+2*c_w)*B - (c_l*y1f+U@ey1)*d1(B) - (c_l*y2f+U@ey2)*d2(B)
    resid=lambda:(float(np.abs(RES_OM.evaluate()["g"]).max()),
                  float(np.abs(RES_B.evaluate()["g"]).max()))
    prob=d3.NLBVP([Pt,Ot,Bt,c_l,c_w,t1,t2,t3,t4,s1,q1],namespace=locals())
    prob.add_equation("lap(Wp*Pt)+lift1(t1,-1)+lift1(t2,-2)+lift2(t3,-1)+lift2(t4,-2)+Wo*Ot = 0")
    prob.add_equation("Pt(y1=0) = 0")
    prob.add_equation("Pt(y2=0) = 0")
    prob.add_equation("d1(Pt)(y1=Ybox) = 0")     # tilde is BOUNDED -> flat far field
    prob.add_equation("d2(Pt)(y2=Ybox) = 0")
    prob.add_equation("c_w*Wo*Ot + d1(Wb*Bt) - (c_l*y1f+U@ey1)*d1(Wo*Ot)"
                      " - (c_l*y2f+U@ey2)*d2(Wo*Ot) + L1(s1,-1) = 0")
    prob.add_equation("(c_l+2*c_w)*Wb*Bt - (c_l*y1f+U@ey1)*d1(Wb*Bt)"
                      " - (c_l*y2f+U@ey2)*d2(Wb*Bt) + L1(q1,-1) = 0")
    prob.add_equation("Ot(y1=0) = 0")
    prob.add_equation("d1(Bt)(y1=0) = 0")
    prob.add_equation("integ(wt*(Wo*Ot)**2) = E1")
    prob.add_equation("integ(wt*r2f*(Wo*Ot)**2) = E2")
    solver=prob.build_solver()
    r0=resid(); print(f"  SEED residual |R_Om|={r0[0]:.4e} |R_B|={r0[1]:.4e}")
    hist=[]
    for it in range(1,iters+1):
        solver.newton_iteration(damping=damping)
        ro,rb=resid()
        cl=float(np.ravel(c_l["g"])[0]); cw=float(np.ravel(c_w["g"])[0])
        g=-cl/cw if abs(cw)>1e-300 else float("nan")
        aO=float(np.abs((Wo*Ot).evaluate()["g"]).max())
        aB=float(np.abs((Wb*Bt).evaluate()["g"]).max())
        hist.append(dict(it=it,res_Om=ro,res_B=rb,c_l=cl,c_w=cw,gamma=g,amp_Om=aO,amp_B=aB))
        print(f"  {it:>3} |R_Om|={ro:.3e} |R_B|={rb:.3e} c_l={cl:+.5f} c_w={cw:+.5f} "
              f"g={g:+.4f} ||Om||={aO:.3e} ||B||={aB:.3e}",flush=True)
        if not all(np.isfinite(v) for v in (ro,rb,cl,cw)): print("  non-finite"); break
        if max(ro,rb)<1e-10: print("  converged"); break
    pathlib.Path(out).parent.mkdir(parents=True,exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(dict(N=N,Ybox=Ybox,history=hist,
        target=dict(c_l=C_L_T,c_w=C_W_T,gamma=GAMMA_T)),indent=2))
    if hist:
        f=hist[-1]; print(f"\n[WEIGHTED] c_l={f['c_l']:.6f} (t {C_L_T})  "
                          f"c_w={f['c_w']:.6f} (t {C_W_T})  gamma={f['gamma']:.6f} (t {GAMMA_T})")
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--N",type=int,default=32); ap.add_argument("--Ybox",type=float,default=8.0)
    ap.add_argument("--iters",type=int,default=20); ap.add_argument("--damping",type=float,default=0.4)
    ap.add_argument("--out",default="../runs/profile_weighted.json")
    a=ap.parse_args(); main(N=a.N,Ybox=a.Ybox,iters=a.iters,damping=a.damping,out=a.out)
