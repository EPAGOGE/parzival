import scipy.io as sio, numpy as np, os
np.set_printoptions(precision=8, suppress=False, linewidth=200, threshold=20)
p2 = os.path.expanduser("~/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat")
d = sio.loadmat(p2, struct_as_record=False, squeeze_me=True)
M = d['Mesh']
x1 = np.asarray(M.x[0]).ravel(); x2 = np.asarray(M.x[1]).ravel()
print("x1 size",x1.size,"x2 size",x2.size)
print("x1 tail:", x1[-6:])
print("max r  =", x1[-1], "log10 =", np.log10(x1[-1]))
print("first nonzero", x1[1], "log10", np.log10(x1[1]))
print("h (Mesh.h)", M.h, "ext", M.ext, "xag shape", np.shape(M.xag))
print("rec =", np.asarray(d['rec']).ravel())
s = d['solu']
for f in s._fieldnames:
    v = getattr(s,f)
    sz = np.size(v)
    print("  solu.",f, np.shape(v), (np.asarray(v).ravel() if sz<6 else ""))
