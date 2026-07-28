import numpy as np
# --- A's shell table on temp600 (640x640) ---
x1=np.load("t600_x1.npy"); w=np.load("t600_w.npy"); cl,cw=np.load("t600_clcw.npy"); al=abs(cw/cl)
A=[1.746766169676,1.591782158291e-03,0.6216578929,-8.246377283470e-03,0.151770113511]
def gw(b,a1,a2,a3,a4,a5):
    z=np.pi/2-b
    return a1*z*(1+a5*z**2)/(((z**2+a2)**(2/3))+a3*z**2+a4)
def chi1(r,a=10.0,l1=50000.0):
    X=(r-a)/np.sqrt(l1)
    out=np.zeros_like(r); m=r>=a
    out[m]=X[m]**7/(1+X[m]**2)**3.5
    return out
sr=np.sqrt(x1[:,None]**2+x1[None,:]**2)
sb=np.arctan(np.divide(x1[None,:],x1[:,None],out=np.full((len(x1),len(x1)),np.nan),where=x1[:,None]!=0))
sb[0,:]=np.pi/2
W1=chi1(sr)*sr**(-al)*gw(sb,*A)
res=w-W1
print("A's shell table (chi1 only, temp600 640x640):")
for lo,hi in [(1e2,1e3),(1e3,1e4),(1e4,1e6),(1e6,1e8),(1e8,1e10),(1e10,1e16)]:
    m=(sr>=lo)&(sr<hi)
    if m.sum()==0: continue
    a_=np.abs(w[m]).max(); b_=np.abs(res[m]).max()
    print("  %.0e-%.0e : max|w|=%.3e  max|w-W1|=%.3e  ratio %.3f  (npts %d)"%(lo,hi,a_,b_,b_/a_,m.sum()))
print()
print("chi1 values: chi1(100)=%.4f chi1(224)=%.4f chi1(1e3)=%.4f chi1(1e4)=%.4f chi1(250)=%.4f"
      %tuple(chi1(np.array([100.,223.607,1e3,1e4,250.]))))

# --- B's angular-shape table on the 620 profile ---
x1b=np.load("x1.npy"); x2b=np.load("x2.npy"); wb=np.load("w620.npy"); vb=np.load("v620.npy")
import scipy.io as sio
d=sio.loadmat("/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat",struct_as_record=False,squeeze_me=True)
alb=float(d['solu'].al)
print("\nB's angular-shape deviations (620 profile, ring vs ring at r~1e13):")
def ring(rt):
    i=np.argmin(abs(x1b-rt))
    # sample along the arc of radius x1b[i] by interpolating on x2 for fixed x1? use the diagonal-free approach:
    # take the grid column set with r ~ rt via polar interp
    return i
# build shape as function of beta at fixed r by interpolation over the 2D grid
from scipy.interpolate import RegularGridInterpolator
itp=RegularGridInterpolator((x1b,x2b),wb,bounds_error=False,fill_value=None)
betas=np.linspace(0.0,np.pi/2*0.999,200)
def shape(r):
    pts=np.stack([r*np.cos(betas),r*np.sin(betas)],axis=-1)
    return itp(pts)*r**alb
ref=shape(1e13)
for rt in [1e4,1e6,1e8,1e10,1e11]:
    s=shape(rt)
    print("  r=%.0e : max rel dev vs r=1e13 ring = %.2e"%(rt,np.abs(s/ref-1).max()))
# decay exponent of wall-ray deviation
r=x1b; g=wb[:,0]*r**alb; asym=g[-40]; devw=np.abs(g/asym-1)
gv=vb[:,0]*r**(2*alb); asv=gv[-40]; devv=np.abs(gv/asv-1)
m=(r>1e3)&(r<1e12)
pw=np.polyfit(np.log(r[m]),np.log(devw[m]),1)[0]
pv=np.polyfit(np.log(r[m]),np.log(devv[m]),1)[0]
print("\n  fitted decay exponent of wall-ray deviation: Om %.4f   v %.4f   (-al = %.4f)"%(pw,pv,-alb))
