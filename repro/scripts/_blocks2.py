import sys,os,numpy as np, scipy.linalg as sla
os.chdir("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
sys.path.insert(0,os.getcwd()); import q345
real,S,a,z = q345.spectrum.load_production("A")
nl, nf = real._nl, real.n_f
V = q345.householder2(real.Cg.T)
def Qx(x):                     # Q = H0 H1  applied to a free-space vector
    y = x.copy()
    for j in (1,0):
        v=V[:,j]; y -= 2.0*v*(v@y)
    return y
L = np.load("q345_Lred_A.npy")
n = L.shape[0]
# top singular pair of Lred
v = np.random.default_rng(0).standard_normal(n); v/=np.linalg.norm(v)
for _ in range(400):
    w = L.T@(L@v); v = w/np.linalg.norm(w)
u = L@v; s0=np.linalg.norm(u); u/=s0
def lift(x):
    e=np.zeros(nf); e[2:]=x; return Qx(e)
uf, vf = lift(u), lift(v)
print(f"||L||_2 (quotient) = {s0:.6e}", flush=True)
print(f"  top RIGHT sing vec: A-half mass {np.linalg.norm(vf[:nl])**2:.6f}  B-half {np.linalg.norm(vf[nl:])**2:.6f}", flush=True)
print(f"  top LEFT  sing vec: A-half mass {np.linalg.norm(uf[:nl])**2:.6f}  B-half {np.linalg.norm(uf[nl:])**2:.6f}", flush=True)
# exact block norms of the QUOTIENT operator, via the lifted A-half projector
E = np.zeros((nf,n)); E[2:,:] = np.eye(n)
Qm = np.empty((nf,n))
for j in range(n):   # cheap: two rank-1 updates per column, done blockwise below
    pass
Y = E.copy()
for j in (1,0):
    vv=V[:,j]; Y -= 2.0*np.outer(vv, vv@Y)
mA = (np.linalg.norm(Y[:nl,:],axis=0)**2)      # not a projector; do it right:
PA = Y[:nl,:].T @ Y[:nl,:]                     # = Z0^T P_A Z0, the compressed A-projector
w = sla.eigvalsh(PA)
print(f"  compressed A-projector spectrum in [{w[0]:.3e}, {w[-1]:.3e}] (a projector: 0/1)", flush=True)
PB = np.eye(n) - PA
for nm, X in (("AA",PA@L@PA),("AB",PA@L@PB),("BA",PB@L@PA),("BB",PB@L@PB)):
    print(f"  ||L_{nm}||_2 = {np.linalg.norm(X,2):.6e}", flush=True)
H = 0.5*(L+L.T)
om = sla.eigvalsh(H, subset_by_index=[n-1,n-1])[0]
print(f"  omega(L) = {om:+.6e}   omega/||L|| = {om/s0:.6f}   (pure off-diagonal block => exactly 0.5)", flush=True)
