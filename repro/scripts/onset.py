import numpy as np, scipy.io as sio
p="/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat"
d=sio.loadmat(p, struct_as_record=False, squeeze_me=True)
M=d["Mesh"]; s=d["solu"]
x=np.asarray(M.x[0]).ravel(); al=float(s.al)
W=np.asarray(s.w[0][0]); V=np.asarray(s.v[0][0]); TH=np.asarray(s.th[0][0])
# wall ray x2=0: w*r^al -> c1 asymptote
wall=W[:,0]*x**al
ref=wall[-40]          # deep far field
print("asymptote c1 =", ref)
print(" r            w*r^al      rel dev from asymptote")
for t in [1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e12,1e14]:
    i=np.argmin(abs(x-t)); print("  %.2e   %.8f   %+.3e"%(x[i],wall[i],wall[i]/ref-1))
# same for v and th on wall
wv=V[:,0]*x**(2*al); rv=wv[-40]
wt=TH[:,0]*x**(2*al-1); rt=wt[-40]
print("\n r          v*r^2al reldev      th*r^(2al-1) reldev")
for t in [1e2,1e4,1e6,1e8,1e10,1e12]:
    i=np.argmin(abs(x-t)); print("  %.2e   %+.3e    %+.3e"%(x[i],wv[i]/rv-1, wt[i]/rt-1))
# angular shape: compare G(beta) at two radii along a ring using bilinear sampling
from scipy.interpolate import RegularGridInterpolator
f=RegularGridInterpolator((x,x),W,bounds_error=False,fill_value=None)
bet=np.linspace(0,np.pi/2,60)[:-1]
def ring(r):
    pts=np.stack([r*np.cos(bet), r*np.sin(bet)],axis=-1)
    return f(pts)*r**al
G_ref=ring(1e13)
print("\n ring r      max rel dev of angular shape vs r=1e13")
for t in [1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e11]:
    G=ring(t); print("  %.1e   %.3e"%(t, np.max(np.abs(G-G_ref))/np.max(np.abs(G_ref))))
