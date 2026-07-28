import numpy as np, scipy.io as sio
p="/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat"
d=sio.loadmat(p, struct_as_record=False, squeeze_me=True)
M=d["Mesh"]; s=d["solu"]
x=np.asarray(M.x[0]).ravel()
def get(cell,i,j):
    a=np.asarray(cell[i][j] if isinstance(cell[i],(list,np.ndarray)) else cell[i,j])
    return a
W=np.asarray(s.w[0][0]); TH=np.asarray(s.th[0][0]); V=np.asarray(s.v[0][0])
U1=np.asarray(s.u1[0][0]); U2=np.asarray(s.u2[0][0])
print("shapes", W.shape, TH.shape, V.shape)
print("x[:4]",x[:4], " x[-1]=%.4e"%x[-1])
for nm,A in [("w",W),("th",TH),("v",V),("u1",U1),("u2",U2)]:
    print(f"{nm}: A[0,:5]={A[0,:5]}   A[:5,0]={A[:5,0]}")
print()
# far-field ray checks at r~1e10
al=s.al
i=np.argmin(abs(x-1e10)); print("index for x~1e10:", i, x[i])
# along axis x1=0 -> A[0, j];  along wall x2=0 -> A[i, 0]
print("w[0,i]=",W[0,i], " w[i,0]=",W[i,0])
print("th[0,i]=",TH[0,i]," th[i,0]=",TH[i,0])
print("v[0,i]=",V[0,i]," v[i,0]=",V[i,0])
print("u1[0,i]=",U1[0,i]," u2[i,0]=",U2[i,0])
# th vs x1*v
print("\ncheck th == x1*v ?")
X1=x[:,None]; 
err=np.abs(TH - X1*V); den=np.abs(TH)+1e-300
sel=np.abs(TH)>1e-12
print("max rel err th vs x1*v on |th|>1e-12: %.3e"%np.max(err[sel]/den[sel]))
# angular profile of w at fixed r: measure g(beta) at wall and axis
print("\nangular: fix i (x1) and j (x2) s.t. r fixed ~1e10")
