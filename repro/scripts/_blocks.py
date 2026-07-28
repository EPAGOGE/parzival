import sys,os,time,numpy as np, scipy.linalg as sla
os.chdir("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
sys.path.insert(0,os.getcwd()); import q345
real,S,a,z = q345.spectrum.load_production("A")
M,_,_ = q345.dense_M_blocked(real)
Ld = M - real.Bc @ np.linalg.solve(real.CgBc, real.Cg @ M); del M
nl = real._nl
blk = {"AA":Ld[:nl,:nl],"AB":Ld[:nl,nl:],"BA":Ld[nl:,:nl],"BB":Ld[nl:,nl:]}
print("block spectral norms of L (free coords, A-half = first %d):"%nl, flush=True)
for k,v in blk.items():
    print(f"   ||L_{k}||_2 = {np.linalg.norm(v,2):.6e}", flush=True)
nL = np.linalg.norm(Ld,2)
print(f"   ||L||_2 = {nL:.6e}", flush=True)
H = 0.5*(Ld+Ld.T)
w = sla.eigvalsh(H, subset_by_index=[Ld.shape[0]-1,Ld.shape[0]-1])[0]
print(f"   omega(L) = {w:+.6e}   omega/||L|| = {w/nL:.6f}   (nilpotent-block prediction 0.5)", flush=True)
# where does the top singular direction live?
u,s,vt = None,None,None
v = np.random.default_rng(0).standard_normal(Ld.shape[0]); v/=np.linalg.norm(v)
for _ in range(200):
    y = Ld@v; w2 = Ld.T@y; v = w2/np.linalg.norm(w2)
u = Ld@v; u/=np.linalg.norm(u)
print(f"   top right sing vec mass: A-half {np.linalg.norm(v[:nl])**2:.6f}  B-half {np.linalg.norm(v[nl:])**2:.6f}", flush=True)
print(f"   top left  sing vec mass: A-half {np.linalg.norm(u[:nl])**2:.6f}  B-half {np.linalg.norm(u[nl:])**2:.6f}", flush=True)
