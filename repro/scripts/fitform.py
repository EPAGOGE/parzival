import numpy as np, scipy.io as sio
from scipy.optimize import curve_fit
p="/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat"
d=sio.loadmat(p, struct_as_record=False, squeeze_me=True)
M=d["Mesh"]; s=d["solu"]
x=np.asarray(M.x[0]).ravel(); al=float(s.al)
W=np.asarray(s.w[0][0]); V=np.asarray(s.v[0][0])
print("al =", al, " ag_coe shapes:", [np.shape(a) for a in s.ag_coe], " xag n=",len(np.asarray(M.xag)))
n=len(x)
# fit window as in Profile_remesh (uses 640 map-mesh); here use the analogous 620 window
r_st=x[n-81]; r_ed=x[n-31]
print("window r_st=%.4e r_ed=%.4e"%(r_st,r_ed))
R=np.sqrt(x[:,None]**2+x[None,:]**2)
B=np.arctan2(x[None,:],x[:,None])          # beta = atan(x2/x1)
sel=(R>r_st)&(R<r_ed)
fx=B[sel]; fy=(W*R**al)[sel]
def form(xb,a1,a2,a3,a4,a5):
    z=np.pi/2-xb
    return a1*z*(1+a5*z**2)/((z**2+a2)**(2/3)+a3*z**2+a4)
p0=[0.727518672351635,0.0306944908727312,0.148981240378485,0.772048460802553,0.299390592212213]
popt,_=curve_fit(form,fx,fy,p0=p0,maxfev=200000, bounds=([-np.inf,0,-np.inf,-np.inf,-np.inf],[np.inf]*5))
res=fy-form(fx,*popt); ss=1-np.sum(res**2)/np.sum((fy-fy.mean())**2)
print("W fit A =", popt, " R^2=%.8f"%ss)
# the wall-matching correction on A(3)
tt=(W[:,0]*x**al); m=(x>r_st)&(x<r_ed); c1=tt[m].mean()
A=popt.copy()
A3new=(A[0]*np.pi/2*(1+A[4]*(np.pi/2)**2)/c1 - A[3] - ((np.pi/2)**2+A[1])**(2/3))/(np.pi/2)**2
print("c1 (wall value of w*r^al) = %.8f   a3_fit=%.6f  a3_corrected=%.6f"%(c1,A[2],A3new))
print("G(beta=0) with corrected a3 = %.8f  (should equal c1)"%form(0.0,A[0],A[1],A3new,A[3],A[4]))
print("G(beta=pi/2) = %.3e (axis, must be 0)"%form(np.pi/2,*popt))
# v fit
fyv=(V*R**(2*al))[sel]
def formv(xb,a1,a2,a3,a4,a5):
    c=np.cos(xb)
    return c*a1*(1+a5*np.sin(xb))/((c**2+a2)**(2/3)+a3+a4*c**2)
p0v=[0.369246781120215,0.111202755293787,0.780252068321138,0.389738836961253,0.241691285913833]
pv,_=curve_fit(formv,fx,fyv,p0=p0v,maxfev=400000,bounds=([-np.inf,0,-np.inf,-np.inf,-1],[np.inf,np.inf,np.inf,np.inf,1]))
resv=fyv-formv(fx,*pv); ssv=1-np.sum(resv**2)/np.sum((fyv-fyv.mean())**2)
print("V fit A =", pv, " R^2=%.8f"%ssv)
print("Gv(beta=0)=%.6f  wall measured mean=%.6f"%(formv(0.0,*pv), (V[:,0]*x**(2*al))[m].mean()))
# where does the power law hold? measure local slope of w on the wall ray
wall=W[:,0]; ls=np.log(np.abs(wall[1:])); lx=np.log(x[1:])
slope=np.gradient(ls,lx)
for target in [1e2,1e4,1e6,1e8,1e10,1e12,1e14]:
    i=np.argmin(abs(x-target)); print("  r=%.1e  local d ln w/d ln r = %+.5f   (al target %+.5f)"%(x[i],slope[i-1],-al))
