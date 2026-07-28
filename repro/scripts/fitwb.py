import numpy as np
from scipy.optimize import curve_fit
def run(tag, x1, x2, w, v, cl, cw, r_st, r_ed):
    al=abs(cw/cl)
    n1,n2=w.shape
    sr=np.sqrt(x1[:,None]**2+x2[None,:]**2)
    sb=np.arctan(x2[None,:]/np.where(x1[:,None]==0,np.nan,x1[:,None]))
    sb[0,:]=np.pi/2
    m=(sr>r_st)&(sr<r_ed)
    X=sb[m]; Y=(w*sr**al)[m]
    o=np.argsort(X); X=X[o]; Y=Y[o]
    def mod(x,a1,a2,a3,a4,a5):
        z=np.pi/2-x
        return a1*z*(1+a5*z**2)/(((z**2+a2)**(2/3))+a3*z**2+a4)
    p0=[0.727518672351635,0.0306944908727312,0.148981240378485,0.772048460802553,0.299390592212213]
    lb=[-np.inf,0,-np.inf,-np.inf,-np.inf]; ub=[np.inf]*5
    A,_=curve_fit(mod,X,Y,p0=p0,bounds=(lb,ub),maxfev=200000)
    res=Y-mod(X,*A); ss=1-np.sum(res**2)/np.sum((Y-Y.mean())**2)
    tt=w[:,0]*x1**al; tt=tt[(x1>r_st)&(x1<r_ed)]; c1=tt.mean()
    A3=(A[0]*np.pi/2*(1+A[4]*(np.pi/2)**2)/c1 - A[3] - ((np.pi/2)**2+A[1])**(2/3))/(np.pi/2)**2
    Ac=A.copy(); Ac[2]=A3
    print("[%s] npts=%d  window r=[%.4e,%.4e]"%(tag,len(X),r_st,r_ed))
    print("   A  = [%.12f, %.12e, %.10f, %.12e, %.12f]"%tuple(A))
    print("   a3 -> %.10f  after wall correction (c1=%.16f)"%(A3,c1))
    print("   R^2 = %.8f   maxabs err = %.4e  peak=%.4f  (%.2f%%)"%(ss,np.abs(res).max(),Y.max(),100*np.abs(res).max()/Y.max()))
    print("   model at beta=0 : %.10f   (target c1 %.10f)"%(mod(0.0,*Ac),c1))
    print("   model at beta=pi/2 : %.3e"%mod(np.pi/2,*Ac))
    # theta / v fit
    Yv=(v*sr**(2*al))[m][o]
    def modv(x,a1,a2,a3,a4,a5):
        cb=np.cos(x)
        return cb*a1*(1+a5*np.sin(x))/(((cb**2+a2)**(2/3))+a3+a4*cb**2)
    p0v=[0.369246781120215,0.111202755293787,0.780252068321138,0.389738836961253,0.241691285913833]
    Av,_=curve_fit(modv,X,Yv,p0=p0v,bounds=([-np.inf,0,-np.inf,-np.inf,-1],[np.inf]*4+[1]),maxfev=200000)
    rv=Yv-modv(X,*Av); ssv=1-np.sum(rv**2)/np.sum((Yv-Yv.mean())**2)
    tv=v[:,0]*x1**(2*al); tv=tv[(x1>r_st)&(x1<r_ed)]
    print("   Av = [%.10f, %.6e, %.10e, %.10f, %.12f]"%tuple(Av))
    print("   R^2v=%.9f  maxerr=%.3e peak=%.4f (%.2f%%)  model@wall=%.7f  measured=%.7f  model@pi/2=%.2e"
          %(ssv,np.abs(rv).max(),Yv.max(),100*np.abs(rv).max()/Yv.max(),modv(0.0,*Av),tv.mean(),modv(np.pi/2,*Av)))
    print()

x1=np.load("t600_x1.npy"); w=np.load("t600_w.npy"); v=np.load("t600_v.npy")
cl,cw=np.load("t600_clcw.npy")
run("temp600 / Profile_remesh window (ANSWER A)", x1, x1, w, v, cl, cw, x1[-81], x1[-31])

x1b=np.load("x1.npy"); x2b=np.load("x2.npy"); wb=np.load("w620.npy"); vb=np.load("v620.npy")
import scipy.io as sio
d=sio.loadmat("/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat",struct_as_record=False,squeeze_me=True)
clb=float(d['solu'].cl); cwb=float(d['solu'].cw)
run("620 profile, B's window [3.6e11,1.04e14]", x1b, x2b, wb, vb, clb, cwb, 3.6e11, 1.04e14)
