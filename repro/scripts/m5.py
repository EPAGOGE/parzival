import scipy.io as sio, numpy as np, os
p2 = os.path.expanduser("~/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat")
d = sio.loadmat(p2, struct_as_record=False, squeeze_me=True)
M=d['Mesh']; x1=np.asarray(M.x[0]).ravel(); x2=np.asarray(M.x[1]).ravel()
s=d['solu']
def cell(f):
    a=getattr(s,f)
    return np.array(a,dtype=object)
for f in ['w','th','v','v_dx','u1','u1_dx','u2']:
    a=cell(f); print(f, "cell shape", a.shape)
W=cell('w'); TH=cell('th'); V=cell('v'); U1=cell('u1'); U2=cell('u2')
def g(C,i,j):
    return np.asarray(C[i,j])
for nm,C in (('w',W),('th',TH),('v',V),('u1',U1),('u2',U2)):
    A=g(C,0,0)
    print(f"--- {nm}{A.shape}: [0,0]={A[0,0]:.6e}  row0(x1=0) max|.|={np.abs(A[0,:]).max():.3e}  col0(x2=0) max|.|={np.abs(A[:,0]).max():.3e}")
# exponents along x1 at fixed x2 index
j=200
i1,i2=480,520
for nm,C in (('w',W),('th',TH),('v',V),('u1',U1),('u2',U2)):
    A=g(C,0,0)
    p=np.log(abs(A[i2,j])/abs(A[i1,j]))/np.log(x1[i2]/x1[i1])
    print(f"  slope {nm} = {p:.5f}")
print("alpha=",s.al, " expect: w->-al=%.5f th->1-2al=%.5f v->-2al=%.5f"%(-s.al,1-2*s.al,-2*s.al))
print("th 1+2*(-al) =",1+2*(-s.al))
# normalization check
print("wx0 = w{2,1}[0,0] =", g(W,1,0)[0,0], " vx0 = v{2,1}[0,0] =", g(V,1,0)[0,0])
print("u1_dx[0,0] =", np.asarray(s.u1_dx)[0,0])
cl=4*g(V,1,0)[0,0]/g(W,1,0)[0,0]; print("cl computed",cl,"stored",s.cl)
print("cw computed", np.asarray(s.u1_dx)[0,0]+cl/2, "stored", s.cw)
# is v = th/x1 or th_x?
th=g(TH,0,0); v=g(V,0,0); thx=g(TH,1,0)
with np.errstate(divide='ignore',invalid='ignore'):
    r1=th[1:,200]/(x1[1:]*v[1:,200])
    r2=thx[1:,200]/v[1:,200]
print("median th/(x1*v) =",np.nanmedian(r1[:400]), " median th_x/v =",np.nanmedian(r2[:400]))
