import scipy.io as sio, numpy as np, os
p2 = os.path.expanduser("~/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat")
d = sio.loadmat(p2, struct_as_record=False, squeeze_me=True)
M=d['Mesh']; x1=np.asarray(M.x[0]).ravel(); x2=np.asarray(M.x[1]).ravel()
s=d['solu']; al=float(s.al)
W=np.array(getattr(s,'w'),dtype=object); V=np.array(getattr(s,'v'),dtype=object)
w=np.asarray(W[0,0]); v=np.asarray(V[0,0])
def slopes(f, j, label):
    print(f"\n-- {label}: local slope of ln|f| vs ln r (target {-al if 'w' in label else -2*al:.5f})")
    for i in [100,150,200,250,300,350,400,450,467,500,540,560,580]:
        r0,r1_=x1[i],x1[i+4]
        a0,a1_=abs(f[i,j]),abs(f[i+4,j])
        if a0==0 or a1_==0: continue
        p=np.log(a1_/a0)/np.log(r1_/r0)
        print(f"   r={r0:9.3e}  slope={p:+.6f}   err={p-(-al if 'w' in label else -2*al):+.2e}")
slopes(w,0,"w  (beta=0, wall)")
slopes(v,0,"v  (beta=0, wall)")
# along the diagonal-ish: fixed x2 index equal to x1 index -> beta=pi/4
print("\n-- w along the diagonal beta=pi/4 (i=j):")
for i in [200,250,300,350,400,450,500,540,560]:
    r0=np.hypot(x1[i],x2[i]); r1_=np.hypot(x1[i+4],x2[i+4])
    p=np.log(abs(w[i+4,i+4])/abs(w[i,i]))/np.log(r1_/r0)
    print(f"   r={r0:9.3e}  slope={p:+.6f}  err={p+al:+.2e}")
