#!/usr/bin/env python3
"""sigma_PEAK: Lambda restricted to the 2x-HWHM box at the vorticity peak.
THE script for the viscous-inversion numbers (S1). Gates: longest clean
stretch of spectral tail <=1e-6 and gamma_drift <=1e-4; 10x-jump trim."""
import glob, h5py, json, numpy as np, sys
sys.path.insert(0,'.')
from lambda_geom import axes, vorticity, grad_xi_sq
def window(tag):
    r=[json.loads(l) for l in open(f"../runs/stream_{tag}.jsonl") if l.strip()]
    t=np.array([x["t"] for x in r])
    ok=np.array([max(x.get("tail_u1",0),x.get("tail_w1",0))<=1e-6
                 and x.get("gamma_drift",0)<=1e-4 for x in r])
    best=(0,0); i=0
    while i<len(ok):
        if ok[i]:
            j=i
            while j<len(ok) and ok[j]: j+=1
            if j-i>best[1]-best[0]: best=(i,j)
            i=j
        else: i+=1
    return (t[best[0]],t[best[1]-1]) if best[1]>best[0] else (0,0)
def ext(prof,i0):
    pk=prof[i0]; h=pk/2; n=len(prof); j=i0
    while j-i0<n-1 and prof[(j+1)%n]>h: j+=1
    k=i0
    while i0-k<n-1 and prof[(k-1)%n]>h: k-=1
    return i0-2*(i0-k), i0+2*(j-i0)
def sigma(tag):
    t0,t1=window(tag); rows=[]
    for fn in sorted(glob.glob(f"../runs/snap_{tag}/*.h5")):
        with h5py.File(fn,'r') as f:
            z,r=axes(f); W,U,st=f["tasks"]["omega1"][:],f["tasks"]["u1"][:],f["scales/sim_time"][:]
            nz,nr=len(z),len(r)
            for n in range(len(st)):
                if st[n]<t0 or st[n]>t1 or st[n]<=0: continue
                wr,wt,wz=vorticity(U[n],W[n],z,r)
                mag=np.sqrt(wr**2+wt**2+wz**2); mx=mag.max()
                s=np.maximum(mag,mx*1e-12)
                g=np.sqrt(np.maximum(grad_xi_sq(wr/s,wt/s,wz/s,z,r),0))
                lam=np.where(mag>0.5*mx,g/np.sqrt(np.maximum(mag,1e-300)),0.0)
                iz,ir=np.unravel_index(np.argmax(mag),mag.shape)
                za,zb=ext(mag[:,ir],iz); ra,rb=ext(mag[iz,:],ir)
                box=np.ix_(np.arange(za,zb+1)%nz,np.arange(max(0,ra),min(nr-1,rb)+1))
                rows.append((mx,float(lam[box].max())))
    out=[rows[0]] if rows else []
    for x in rows[1:]:
        if x[0]>out[-1][0]*10: break
        out.append(x)
    if len(out)<6: return None,len(out)
    a=np.array(out)
    return float(np.polyfit(np.log(a[:,0]),np.log(np.maximum(a[:,1],1e-12)),1)[0]),len(out)
if __name__=="__main__":
    print(f"{'run':<14}{'grid':>10}{'nu':>8}{'n':>4}{'sigma_PEAK':>12}")
    for tag,g,nu in [("OR_z128r384","128x384",0),("NUL1e-4","128x384",1e-4),
                     ("NUL1e-3","128x384",1e-3),("OR_z256r768","256x768",0),
                     ("N2_1e-4","256x768",1e-4),("N2_1e-3","256x768",1e-3)]:
        try: s,n=sigma(tag)
        except FileNotFoundError: continue
        print(f"{tag:<14}{g:>10}{nu:>8.0e}{n:>4}{s if s is None else round(s,4)!s:>12}")
