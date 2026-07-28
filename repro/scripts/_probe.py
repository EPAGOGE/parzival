import sys,os,time,numpy as np
os.chdir("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
sys.path.insert(0,os.getcwd()); import q345
T=np.load("q345_T_A.npy"); ss=q345.SchurSigma(None,T=T); ss.tol=1e-12
ev=np.load("q345_ev_A.npy")
xs=np.linspace(-2,2,81)
row=[]
for x in xs:
    z=complex(x,0.2); s,it=ss.sigma_min(z,warm=False); row.append(s)
row=np.array(row); k=int(np.argmin(row))
print(f"y=0.2 row, tol 1e-12: min sigma_min = {row[k]:.6e} at Re = {xs[k]:+.4f}", flush=True)
o=np.argsort(row)[:6]
for i in sorted(o):
    z=complex(xs[i],0.2); d=np.abs(ev-z); j=int(np.argmin(d))
    print(f"   Re={xs[i]:+.3f}  sigma_min={row[i]:.6e}   nearest eigenvalue {ev[j].real:+.6f}{ev[j].imag:+.6f}i  dist={d[j]:.4e}  ratio dist/sigma={d[j]/row[i]:.4e}", flush=True)
zb=complex(xs[k],0.2)
print(f"\nINDEPENDENT SPARSE ROUTE at z={zb}", flush=True)
real,S,a,zz = q345.spectrum.load_production("A")
t0=time.time(); R=q345.spectrum.Resolvent(real, zb); n_,it_=R.norm(iters=400,tol=1e-11)
print(f"   sparse bordered ||R|| = {n_:.6e}  -> sigma_min = {1/n_:.6e}   [{time.time()-t0:.0f}s, {it_} it]", flush=True)
print(f"   dense Schur   sigma_min = {row[k]:.6e}    rel diff = {abs(1/n_-row[k])/row[k]:.3e}", flush=True)
