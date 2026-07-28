import numpy as np, scipy.io as sio
d = sio.loadmat('/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat',
                struct_as_record=False, squeeze_me=True)
solu, Mesh, rec = d['solu'], d['Mesh'], d['rec']
x1 = np.asarray(Mesh.x[0], float); x2 = np.asarray(Mesh.x[1], float)
def F(n):
    a = getattr(solu, n)
    while not (isinstance(a,np.ndarray) and a.dtype!=object and a.ndim==2): a = a[0] if a.shape else a
    return np.asarray(a,float)
w, th, v = F('w'), F('th'), F('v')
print("shapes", w.shape, len(x1))
print("max|th - x1*v| =", np.max(np.abs(th - x1[:,None]*v)))
print("max|v| =", np.max(np.abs(v)), " max|w|=",np.max(np.abs(w)))
print("solu.al  =", repr(float(solu.al)))
print("solu.cl  =", repr(float(solu.cl)), " solu.cw =", repr(float(solu.cw)))
print("|cw/cl|  =", repr(abs(float(solu.cw)/float(solu.cl))))
print("rec      =", np.asarray(rec,float))
r = np.asarray(rec,float)
print("4*rec3/rec2 =", repr(4*r[3]/r[2]), " vs rec0 =", repr(r[0]))
print("rec4+rec0/2 =", repr(r[4]+r[0]/2), " vs rec1 =", repr(r[1]))
# frozen history constants
print("wbx0 target 1.196203150519860 vs rec[2] =", repr(r[2]), " diff", r[2]-1.196203150519860)
print("vbx0 target 0.899095589986449 vs rec[3] =", repr(r[3]), " diff", r[3]-0.899095589986449)
# symmetry line
print("max|w[0,:]| =", np.max(np.abs(w[0,:])), " max|v[0,:]| =", np.max(np.abs(v[0,:])))
print("max|w[:,0]| (wall) =", np.max(np.abs(w[:,0])))
