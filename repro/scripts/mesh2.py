import numpy as np, scipy.io as sio, os
# ---- Ymesh.m ----
def Ymesh(n, BD, ref, m, r1, m1, r2, m2, r3=None, m3=None):
    h = BD/(n-1)
    Y0 = np.arange(0, n)                     # 0:n-1
    tail = Y0[-ref:]
    Y = np.zeros(n + m*ref)
    Y[:n] = Y0
    for i in range(1, m+1):
        Y[n+(i-1)*ref : n+i*ref] = tail - i/(m+1)
    mid = Y0[-ref-r1:-ref] if r1>0 else np.array([])
    if r1>0:
        mid = mid[None,:] - (np.arange(1,m1+1)[:,None])/(m1+1)
        Y = np.concatenate([Y, mid.reshape(-1)])
    if r2>0:
        hd = Y0[:r2][None,:] + (np.arange(1,m2+1)[:,None])/(m2+1)
        Y = np.concatenate([Y, hd.reshape(-1)])
    if r3 is not None and r3>0:
        mid = Y0[r2:r2+r3][None,:] + (np.arange(1,m3+1)[:,None])/(m3+1)
        Y = np.concatenate([Y, mid.reshape(-1)])
    Y = np.sort(Y)*h
    return Y

def mapf(y, M, d2, M2, c, d=6):
    return c*(M*y + M2*y**d2)/(1-y**2)**d

BD=0.991; n1=600
y1 = Ymesh(n1, BD, 8, 2, 24, 1, 0, 1, 0, 2)
print("len(y1) =", len(y1))
x1 = mapf(y1, 10, 25, 150000, 1)
avg=12
x1[:avg] = np.linspace(x1[0], x1[avg-1], avg)
print("map-mesh: x1[0:4]=", x1[:4])
print("x1[-1] = %.6e  s=%.4f" % (x1[-1], np.log(x1[-1])))
print("x1[-2] = %.6e" % x1[-2])
print("r_st=x1[end-80] = %.6e s=%.4f" % (x1[-81], np.log(x1[-81])))
print("r_ed=x1[end-30] = %.6e s=%.4f" % (x1[-31], np.log(x1[-31])))
print("corner sqrt2*x1max = %.6e s=%.4f" % (np.sqrt(2)*x1[-1], np.log(np.sqrt(2)*x1[-1])))

# 620-pt mesh in Steady_state_pertb_oneMesh62036.mat
p="/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat"
d=sio.loadmat(p, struct_as_record=False, squeeze_me=True)
M=d["Mesh"]; s=d["solu"]
print("\nMesh fields:", M._fieldnames)
print("solu fields:", s._fieldnames)
for f in M._fieldnames:
    v=getattr(M,f)
    print("  Mesh.%-8s %s" % (f, np.shape(v) if hasattr(v,'shape') else v))
xx = M.x
try:
    a=np.asarray(xx[0]).ravel(); b=np.asarray(xx[1]).ravel()
    print("  Mesh.x[0]: n=%d  max=%.6e  s=%.4f" % (len(a), a[-1], np.log(a[-1])))
    print("  Mesh.x[1]: n=%d  max=%.6e" % (len(b), b[-1]))
except Exception as e: print(e)
if hasattr(M,'gx'):
    g=M.gx
    a=np.asarray(g[0]).ravel(); print("  Mesh.gx[0]: n=%d max=%.6e s=%.4f" % (len(a), a[-1], np.log(a[-1])))
for f in s._fieldnames:
    v=getattr(s,f)
    print("  solu.%-10s %s" % (f, np.shape(v) if hasattr(v,'shape') else v))
print("rec =", d["rec"])
