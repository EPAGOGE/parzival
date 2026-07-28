import numpy as np, scipy.io as sio
P="/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/"
d=sio.loadmat(P+"temp600_991_10_15W_824_635.mat", struct_as_record=False, squeeze_me=True)
print("keys:", sorted(k for k in d if not k.startswith("__")))
for k in sorted(d):
    if k.startswith("__"): continue
    a=d[k]
    try: print("  %-12s shape=%s"%(k,np.shape(a)))
    except: pass
cl=float(d['cl']); cw=float(d['cw'])
print("cl = %.16f"%cl); print("cw = %.16f"%cw)
print("cw/cl = %.16f"%(cw/cl)); print("cl+2cw = %.16f"%(cl+2*cw)); print("-cl/cw = %.16f"%(-cl/cw))

# rebuild the 600/640 mesh per Profile_remesh.m
def Ymesh(n,BD,ref,m,r1,m1,r2,m2,r3,m3):
    h=BD/(n-1); Y0=np.arange(n,dtype=float)
    tail=Y0[n-ref:]
    Y=np.zeros(n+m*ref); Y[:n]=Y0
    for i in range(1,m+1):
        Y[n+(i-1)*ref:(n+(i-1)*ref+ref)]=tail-i/(m+1)
    mid=Y0[n-ref-r1:n-ref]
    mid=mid[None,:]-(np.arange(1,m1+1)[:,None])/(m1+1)
    Y=np.concatenate([Y,mid.reshape(-1)])
    hd=Y0[:r2]; hd=hd[None,:]+(np.arange(1,m2+1)[:,None])/(m2+1)
    Y=np.concatenate([Y,hd.reshape(-1)])
    mid=Y0[r2:r2+r3]; mid=mid[None,:]+(np.arange(1,m3+1)[:,None])/(m3+1)
    Y=np.concatenate([Y,mid.reshape(-1)])
    Y=np.sort(Y)*h
    return Y
def mapf(y,M,d2,M2,c,dd=6):
    return c*(M*y+M2*y**d2)/(1-y**2)**dd
BD=0.991; n1=600; M=10; M2=150000; d2=25; c=1
y1=Ymesh(n1,BD,8,2,24,1,0,1,0,2)
x1=mapf(y1,M,d2,M2,c)
avg=12
x1[:avg]=np.linspace(x1[0],x1[avg-1],avg)
print("rebuilt mesh: n=%d  x1_max=%.6e  s=%.4f  corner=%.4e s=%.4f"
      %(len(x1),x1[-1],np.log(x1[-1]),np.sqrt(2)*x1[-1],np.log(np.sqrt(2)*x1[-1])))
w=np.asarray(d['w']); v=np.asarray(d['v'])
print("w.shape",w.shape,"v.shape",v.shape)
al=abs(cw/cl)
r_st=x1[-81]; r_ed=x1[-31]
print("fit annulus r_st=x1(end-80)=%.4e  r_ed=x1(end-30)=%.4e"%(r_st,r_ed))
m=(x1>r_st)&(x1<r_ed)
tt=w[m,0]*x1[m]**al
print("c1 = mean(w(:,1)*x1^al) over annulus = %.16f  spread=%.3e rel=%.2e  npts=%d"
      %(tt.mean(),tt.max()-tt.min(),(tt.max()-tt.min())/tt.mean(),m.sum()))
tv=v[m,0]*x1[m]**(2*al)
print("v wall mean = %.10f  spread=%.3e rel=%.2e"%(tv.mean(),tv.max()-tv.min(),(tv.max()-tv.min())/tv.mean()))
np.save("t600_x1.npy",x1); np.save("t600_w.npy",w); np.save("t600_v.npy",v)
np.save("t600_clcw.npy",np.array([cl,cw]))
