import scipy.io as sio, numpy as np, os
p1 = os.path.expanduser("~/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/temp600_991_10_15W_824_635.mat")
d = sio.loadmat(p1, struct_as_record=False, squeeze_me=True)
print("l0 =", d['l0'], " cl,cw =", d['cl'], d['cw'], " -cw/cl =", -d['cw']/d['cl'])
mw=np.asarray(d['maxw']).ravel(); mv=np.asarray(d['maxv']).ravel()
nz=np.nonzero(mw)[0]
print("maxw len",mw.size,"last nonzero idx",nz[-1] if nz.size else None)
print("maxw[[0,1,10,100,1000,5000,10000,18000,18780,18786]] =", mw[[0,1,10,100,1000,5000,10000,18000,18780,18786]])
print("maxv same =", mv[[0,1,10,100,1000,5000,10000,18000,18780,18786]])
print("Fw max abs =", np.abs(d['Fw']).max(), " interior", np.abs(d['Fw'][:-2,:-2]).max())
print("Fv max abs =", np.abs(d['Fv']).max(), " interior", np.abs(d['Fv'][:-2,:-2]).max())
print("w shape",d['w'].shape)
print("s01/s02/s03 shapes", d['s01'].shape)
