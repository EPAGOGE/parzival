import numpy as np, scipy.io as sio
P="/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/"
d=sio.loadmat(P+"Steady_state_pertb_oneMesh62036.mat", struct_as_record=False, squeeze_me=True)
print("keys:", [k for k in d if not k.startswith("__")])
solu=d['solu']; Mesh=d['Mesh']
print("solu fields:", solu._fieldnames)
print("Mesh fields:", Mesh._fieldnames)
al=float(solu.al); cl=float(solu.cl); cw=float(solu.cw)
print("solu.al = %.16f"%al)
print("solu.cl = %.16f"%cl)
print("solu.cw = %.16f"%cw)
print("|cw/cl|  = %.16f"%abs(cw/cl))
print("diff al vs |cw/cl| = %.3e" % (abs(cw/cl)-al))
print("cl+2cw = %.16f"%(cl+2*cw))
print("-cl/cw = %.16f"%(-cl/cw))
x1=np.asarray(Mesh.x[0]).ravel(); x2=np.asarray(Mesh.x[1]).ravel()
print("Mesh.x{1}: n=%d  max=%.6e   corner=%.4e  s_corner=%.4f"%(len(x1),x1[-1],np.sqrt(2)*x1[-1],np.log(np.sqrt(2)*x1[-1])))
print("Mesh.ext =",Mesh.ext, " Mesh.rate =",Mesh.rate)
xag=np.asarray(Mesh.xag).ravel()
print("Mesh.xag: n=%d min=%.16f max=%.16f (pi/2=%.16f)"%(len(xag),xag.min(),xag.max(),np.pi/2))
gx=Mesh.gx
gx1=np.asarray(gx[0]).ravel()
print("Mesh.gx{1}: n=%d max=%.6e s=%.3f corner=%.4e s=%.3f"%(len(gx1),gx1[-1],np.log(gx1[-1]),np.sqrt(2)*gx1[-1],np.log(np.sqrt(2)*gx1[-1])))
ag=solu.ag_coe
print("ag_coe: n=%d, lengths="%len(ag), [np.asarray(a).size for a in ag])
w=np.asarray(solu.w[0,0]); th=np.asarray(solu.th[0,0]); v=np.asarray(solu.v[0,0])
u1=np.asarray(solu.u1[0,0]); u2=np.asarray(solu.u2[0,0])
print("shapes w,th,v:",w.shape,th.shape,v.shape)
diff=th - x1[:,None]*v
print("max|th - x1*v| = %.3e ; max|th| = %.3e ; rel = %.3e"%(np.abs(diff).max(),np.abs(th).max(),np.abs(diff).max()/np.abs(th).max()))
print("max|w[0,:]|=%.3e  max|th[0,:]|=%.3e  max|v[0,:]|=%.3e  max|u1[0,:]|=%.3e"%(np.abs(w[0]).max(),np.abs(th[0]).max(),np.abs(v[0]).max(),np.abs(u1[0]).max()))
print("max|u2[:,0]|=%.3e ; w[:,0] nonzero? max=%.4f ; v[:,0] max=%.4f"%(np.abs(u2[:,0]).max(),np.abs(w[:,0]).max(),np.abs(v[:,0]).max()))
np.save("x1.npy",x1); np.save("x2.npy",x2); np.save("w.npy",w); np.save("v.npy",v); np.save("th.npy",th)
np.save("al.npy",np.array([al]))
