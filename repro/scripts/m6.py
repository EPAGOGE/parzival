import scipy.io as sio, numpy as np, os
p2 = os.path.expanduser("~/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat")
d = sio.loadmat(p2, struct_as_record=False, squeeze_me=True)
M=d['Mesh']; x1=np.asarray(M.x[0]).ravel()[:,None]; x2=np.asarray(M.x[1]).ravel()[None,:]
s=d['solu']
C=lambda f: np.array(getattr(s,f),dtype=object)
W=C('w'); V=C('v'); U1=C('u1'); U2=C('u2')
w=np.asarray(W[0,0]); wy=np.asarray(W[0,1]); wx=np.asarray(W[1,0])
v=np.asarray(V[0,0]); vy=np.asarray(V[0,1]); vx=np.asarray(V[1,0])
u1=np.asarray(U1[0,0]); u2=np.asarray(U2[0,0]); u1dx1=np.asarray(s.u1_dx)
cl=float(s.cl); cw=float(s.cw)
Fv = -(cl*x1+u1)*vx - (cl*x2+u2)*vy + (2*cw-u1dx1)*v
Fw = -(cl*x1+u1)*wx - (cl*x2+u2)*wy + cw*w + v + x1*vx
print("recomputed residual  max|Fv| interior = %.3e   max|Fw| = %.3e"%(np.abs(Fv[:-2,:-2]).max(), np.abs(Fw[:-2,:-2]).max()))
print("stored               max|Fv| interior = %.3e   max|Fw| = %.3e"%(np.abs(d['Fv'][:-2,:-2]).max(), np.abs(d['Fw'][:-2,:-2]).max()))
print("diff from stored Fv: %.3e   Fw: %.3e"%(np.abs(Fv-d['Fv']).max(), np.abs(Fw-d['Fw']).max()))
print()
print("v[0,:] max|.| =", np.abs(v[0,:]).max(), "  v[1,:5] =", v[1,:5])
print("w[0,:] max|.| =", np.abs(w[0,:]).max())
print("v[:,0] first 5 =", v[:5,0], "  w[:,0] first 5 =", w[:5,0])
print("u1[0,:] max =", np.abs(u1[0,:]).max(), " u2[:,0] max =", np.abs(u2[:,0]).max())
print("u1[:,0] first5 =", u1[:5,0])
print("u2[0,:5] =", u2[0,:5])
# is v odd in x1 near axis: v[i,j]/x1[i] ~ const?
print("v[1:4,300]/x1[1:4,0] =", (v[1:4,300]/x1[1:4,0]))
print("v_dx[0:3,300] =", np.asarray(s.v_dx)[0:3,300])
