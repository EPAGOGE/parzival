import numpy as np
rng = np.random.default_rng(0)

C0, T0, R0 = 1.23, 1.7135, 2.60
t = np.linspace(0.80, 1.53, 40)          # ~40 coefficient-space checkpoints
def model(t, C, T, r): return C*(T-t)**r
d0 = model(t, C0, T0, R0)

def profile_fit(t, y, Tgrid):
    """For each candidate T*, ln-space linear fit gives (lnC, rho) exactly. Return best + SSE curve."""
    ly = np.log(y); out=[]
    for T in Tgrid:
        if T <= t.max()+1e-9: out.append((np.inf,np.nan,np.nan)); continue
        x = np.log(T-t)
        A = np.vstack([np.ones_like(x), x]).T
        c, *_ = np.linalg.lstsq(A, ly, rcond=None)
        sse = float(((ly - A@c)**2).sum())
        out.append((sse, float(np.exp(c[0])), float(c[1])))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out]), np.array([o[2] for o in out])

Tgrid = np.arange(1.535, 3.001, 0.0005)

# --- TEST 1: how distinguishable are rho=1.5 / 1.0 from rho=2.6 INSIDE the fit window? ---
print("=== TEST 1: best-fit alternative rho, forced, vs noiseless rho=2.6 data ===")
for rfix in (1.0, 1.5, 2.0):
    best=None
    for T in Tgrid:
        x=np.log(T-t); ly=np.log(d0)
        lnC = np.mean(ly - rfix*x)               # only C free once rho fixed
        res = ly - (lnC + rfix*x)
        m = np.abs(np.exp(res)-1).max()          # max relative error in delta
        if best is None or m<best[0]: best=(m,T,np.exp(lnC))
    print(f"  rho fixed {rfix}: best T*={best[1]:.4f} C={best[2]:.3f} -> MAX rel. residual in delta = {100*best[0]:.2f}%")
