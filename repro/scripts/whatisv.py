import importlib.util, sys, numpy as np
from scipy.io import loadmat
sys.path.insert(0, ".")
sp = importlib.util.spec_from_file_location("ps", "polar_seed.py")
ps = importlib.util.module_from_spec(sp); sys.modules["ps"] = ps; sp.loader.exec_module(ps)
P = ps.load(); d = loadmat(ps.MAT, squeeze_me=True, struct_as_record=False); s_ = d["solu"]
w = np.asarray(d["w"], float); shp = w.shape
X, Y = P["X"], P["Y"]
G = {nm: ps._grid_field(d, s_, nm, shp) for nm in ("th","u1","u2","v","u1_dx","v_dx")}
th, v = G["th"], G["v"]
thx = np.gradient(th, X, axis=0)
thy = np.gradient(th, Y, axis=1)
# clean interior region, away from origin and outer edge
i0,i1,j0,j1 = 60, 400, 60, 400
sl = (slice(i0,i1), slice(j0,j1))
def rel(a,b):
    return float(np.linalg.norm((a-b)[sl])/max(np.linalg.norm(b[sl]),1e-300))
print("candidate identifications for 'v' (clean interior block):")
for lab, cand in (("theta_x", thx), ("theta_x/2", thx/2), ("2*theta_x", 2*thx),
                  ("theta_y", thy), ("u1", G["u1"]), ("u2", G["u2"])):
    print("   v vs %-12s rel L2 = %.4e" % (lab, rel(v, cand)))
r = (v/np.where(np.abs(thx)>0, thx, np.nan))[sl]
r = r[np.isfinite(r)]
print("\n   pointwise ratio v/theta_x over the block: median=%.6f  10-90pct=[%.4f, %.4f]  std/med=%.3f"
      % (np.median(r), np.percentile(r,10), np.percentile(r,90), r.std()/abs(np.median(r))))
print("   -> a CONSTANT ratio would mean v is a rescaling of theta_x; a varying one means it is not.")
# what does v_dx relate to?
print("\n   v_dx vs d(v)/dy1: rel L2 = %.4e" % rel(G["v_dx"], np.gradient(v, X, axis=0)))
print("   u1_dx vs d(u1)/dy1: rel L2 = %.4e" % rel(G["u1_dx"], np.gradient(G["u1"], X, axis=0)))
# exponents for reference
XX,YY = np.meshgrid(X,Y,indexing="ij"); R=np.sqrt(XX**2+YY**2); B=np.arctan2(YY,XX)
def ex(F,b0=0.8):
    m=(np.abs(B-b0)<0.02)&(R>1e8)&(R<1e15)&(np.abs(F)>0)
    return np.polyfit(np.log(R[m]),np.log(np.abs(F[m])),1)[0]
print("\n   exponents at beta=0.8:  v=%.5f  theta_x=%.5f  theta_y=%.5f  th=%.5f"
      % (ex(v), ex(thx), ex(thy), ex(th)))
