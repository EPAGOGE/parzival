import scipy.io as sio, numpy as np, os
p2 = os.path.expanduser("~/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat")
d = sio.loadmat(p2, struct_as_record=False, squeeze_me=True)
M = d['Mesh']
x1 = np.asarray(M.x[0]).ravel(); x2 = np.asarray(M.x[1]).ravel()
print("x1.size",x1.size,"x2.size",x2.size)
print("x1[:4]",x1[:4]); print("x1[-4:]",x1[-4:])
print("outer radius x1[-1] = %.6e  log10=%.3f"%(x1[-1],np.log10(x1[-1])))
print("Mesh.h",np.asarray(M.h).ravel(),"Mesh.ext",M.ext)
r = x1[1:]/x1[:-1]
print("max ratio",r[1:].max(),"ratio at tail",r[-5:])
# where does |x1| cross 1e8?
i=np.searchsorted(x1,1e8); print("index of r=1e8:",i,"of",x1.size)
print("rec",np.asarray(d['rec']).ravel())
# solu.v vs th exponents
s=d['solu']
w=np.asarray(s.w); th=np.asarray(s.th); v=np.asarray(s.v)
print("shapes",w.shape,th.shape,v.shape)
# measure exponent along x1 at fixed x2 index mid
j=300
for name,f in (("w",w),("th",th),("v",v)):
    a=np.asarray(f)
    if a.ndim!=2: print(name,"ndim",a.ndim); continue
    i1,i2=500,540
    p=np.log(abs(a[i2,j])/abs(a[i1,j]))/np.log(x1[i2]/x1[i1])
    print(f"  {name}: local slope d ln|f| / d ln r = {p:.5f}   (x1 {x1[i1]:.3e}->{x1[i2]:.3e})")
