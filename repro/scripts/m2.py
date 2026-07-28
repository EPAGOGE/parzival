import scipy.io as sio, numpy as np, os
p2 = os.path.expanduser("~/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat")
d = sio.loadmat(p2, struct_as_record=False, squeeze_me=True)
M = d['Mesh']; print("Mesh fields:", M._fieldnames)
for f in M._fieldnames:
    v = getattr(M,f)
    print(" ",f, np.shape(v), (v if np.size(v)<12 else ""))
x = M.x
print("type x", type(x), np.shape(x))
x1 = x[0] if isinstance(x,(list,np.ndarray)) else x
x1 = np.asarray(x1).ravel()
print("x1 len", x1.size, "x1[0:5]", x1[:5], "x1[-5:]", x1[-5:])
print("max r =", x1[-1], " log10 =", np.log10(x1[-1]))
print("rec =", d['rec'])
s = d['solu']; print("solu fields:", s._fieldnames)
for f in s._fieldnames:
    v = getattr(s,f)
    print("  solu.",f, np.shape(v), (v if np.size(v)<6 else ""))
p1 = os.path.expanduser("~/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/temp600_991_10_15W_824_635.mat")
d1 = sio.loadmat(p1, struct_as_record=False, squeeze_me=True)
print("temp600 cl,cw =", d1['cl'], d1['cw'], " -cw/cl =", -d1['cw']/d1['cl'], " -cl/cw=", -d1['cl']/d1['cw'])
print("temp600 l0 =", d1['l0'], "maxw last 3", d1['maxw'][-3:], "maxv last3", d1['maxv'][-3:])
