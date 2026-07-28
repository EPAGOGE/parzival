import numpy as np, scipy.io as sio
P="/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/"
d=sio.loadmat(P+"Steady_state_pertb_oneMesh62036.mat", struct_as_record=False, squeeze_me=True)
solu=d['solu']; Mesh=d['Mesh']
x1=np.asarray(Mesh.x[0]).ravel(); x2=np.asarray(Mesh.x[1]).ravel()
al=float(solu.al)
ag=solu.ag_coe
print("ag_coe entries:", len(ag), [np.asarray(a).size for a in ag])
w=np.asarray(solu.w[0,0]); th=np.asarray(solu.th[0,0]); v=np.asarray(solu.v[0,0])
u1=np.asarray(solu.u1[0,0]); u2=np.asarray(solu.u2[0,0])
print("shapes:",w.shape,th.shape,v.shape)
diff=th - x1[:,None]*v
print("max|th - x1*v| = %.3e ; max|th| = %.3e"%(np.abs(diff).max(),np.abs(th).max()))
print("axis x1=0: max|w[0,:]|=%.3e max|th[0,:]|=%.3e max|v[0,:]|=%.3e max|u1[0,:]|=%.3e"
      %(np.abs(w[0]).max(),np.abs(th[0]).max(),np.abs(v[0]).max(),np.abs(u1[0]).max()))
print("wall x2=0: max|u2[:,0]|=%.3e ; max|w[:,0]|=%.4f ; max|v[:,0]|=%.4f"
      %(np.abs(u2[:,0]).max(),np.abs(w[:,0]).max(),np.abs(v[:,0]).max()))
# theta_x vs v: is v == theta_x ?
thx=np.asarray(solu.th[1,0])
print("max|v - th_x| = %.3e  (max|v|=%.3e) -> v is NOT theta_x" % (np.abs(v-thx).max(), np.abs(v).max()))
print("check th_x == v + x1*v_x :", np.abs(thx-(v+x1[:,None]*np.asarray(solu.v[1,0]))).max())
# wall trace c1 over the fit window used in Profile_remesh style (end-80..end-30 of THIS mesh)
for (lo,hi,lab) in [(len(x1)-81,len(x1)-31,"x1(end-80..end-30)"),(len(x1)-61,len(x1)-13,"x1(end-60..end-12)")]:
    tt=w[lo:hi,0]*x1[lo:hi]**al
    print("%s r=[%.4e,%.4e]  mean w*r^al = %.16f  spread %.3e (rel %.2e)"
          %(lab,x1[lo],x1[hi-1],tt.mean(),tt.max()-tt.min(),(tt.max()-tt.min())/tt.mean()))
    tv=v[lo:hi,0]*x1[lo:hi]**(2*al)
    print("     mean v*r^2al = %.10f  spread %.3e (rel %.2e)"%(tv.mean(),tv.max()-tv.min(),(tv.max()-tv.min())/tv.mean()))
# onset diagnostics along wall ray
r=x1
def dev(f,p):
    g=f[:,0]*r**p
    asym=g[-40]
    return g/asym-1
gw=dev(w,al); gv=dev(v,2*al)
for target in [1e2,1e4,1e6,1e8,1e10]:
    i=np.argmin(abs(r-target))
    print("  r=%.1e : (w*r^al)/asym-1 = %+.2e ; (v*r^2al)/asym-1 = %+.2e"%(r[i],gw[i],gv[i]))
np.save("x1.npy",x1); np.save("x2.npy",x2); np.save("w620.npy",w); np.save("v620.npy",v)
