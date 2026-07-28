import numpy as np, scipy.io as sio
p="/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat"
d=sio.loadmat(p, struct_as_record=False, squeeze_me=True)
M=d["Mesh"]; s=d["solu"]
x=np.asarray(M.x[0]).ravel(); al=float(s.al)
W=np.asarray(s.w[0][0]); V=np.asarray(s.v[0][0])
for nm,A,e in [("w",W,al),("v",V,2*al)]:
    q=A[:,0]*x**e; ref=q[-30]
    dev=np.abs(q/ref-1)
    m=(x>1e3)&(x<1e13)&(dev>0)
    sl=np.polyfit(np.log(x[m]),np.log(dev[m]),1)
    print("%s wall-ray: |dev| ~ r^(%.4f)   (al=%.4f)"%(nm,sl[0],al))
