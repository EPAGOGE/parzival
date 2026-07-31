#!/usr/bin/env python3
"""BACKWARDS PROOF. Recomputes every load-bearing number from raw inputs.
No prose. Three tiers: ARITH (pure arithmetic from stated inputs),
EXACT (reference value forced by an exact identity of the PDE, zero free
parameters), FIT (regression; labeled, attackable). PASS = |val-ref|<=tol."""
import glob, h5py, json, hashlib, numpy as np, sys
sys.path.insert(0,'.')
from lambda_geom import axes, vorticity, grad_xi_sq

R=[]
def ck(name,kind,val,ref,tol): R.append((name,kind,val,ref,tol,abs(val-ref)<=tol))

# ---------- TIER A: arithmetic. Inputs: alpha (ours), alpha_1..3 (published).
al=-0.34240; br=[al,-0.4168236,-0.4439811,-0.4578230]
ck("c_l = -1/alpha","ARITH",-1/al,2.9205607,1e-6)
ck("Re exponent -1+2c_l","ARITH",-1+2*(-1/al),4.8411215,1e-6)
d=np.diff(br)
ck("branch step ratio d2/d1","ARITH",d[1]/d[0],0.3649,1e-3)
ck("branch step ratio d3/d2","ARITH",d[2]/d[1],0.5097,1e-3)
ck("Aitken limit alpha_inf","ARITH",br[3]-d[2]**2/(d[2]-d[1]),-0.4722,1e-3)
ck("alpha needed for c_l<1/2","ARITH",-1/0.5,-2.0,1e-12)

# ---------- TIER B: exact identities on raw snapshot bytes.
def gate_t(tag):
    """FIT-tier protocol, part of the claim: trust only records with spectral
    tail <= 1e-6 (longest clean stretch), then trim at any 10x jump."""
    try: rs=[json.loads(l) for l in open(f"../runs/stream_{tag}.jsonl") if l.strip()]
    except Exception: return None
    bad=[x["t"] for x in rs if max(x.get("tail_u1",0),x.get("tail_w1",0))>1e-6]
    return min(bad) if bad else None

def series(tag, gated=False):
    T=gate_t(tag) if gated else None
    out=[]
    for fn in sorted(glob.glob(f"../runs/snap_{tag}/*.h5")):
        with h5py.File(fn,'r') as f:
            z,r=axes(f); W,U,st=f["tasks"]["omega1"][:],f["tasks"]["u1"][:],f["scales/sim_time"][:]
            for n in range(len(st)):
                if st[n]<=0 or (T is not None and st[n]>T): continue
                wr,wt,wz=vorticity(U[n],W[n],z,r)
                mag=np.sqrt(wr**2+wt**2+wz**2); mx=mag.max()
                s=np.maximum(mag,mx*1e-12)
                g=np.sqrt(np.maximum(grad_xi_sq(wr/s,wt/s,wz/s,z,r),0))
                m=mag>0.5*mx
                if m.sum()<8: continue
                out.append((mx,float((g[m]/np.sqrt(mag[m])).max()),float(g[m].max())))
    trim=[out[0]] if out else []
    for x in out[1:]:
        if x[0]>trim[-1][0]*10: break
        trim.append(x)
    return np.array(trim)
A=series("SYMa"); B=series("SYMb")
ck("IC linearity ||w||0(2A)/||w||0(A)","EXACT",B[0,0]/A[0,0],2.0,1e-3)
gA=np.log(A[:,0]/A[0,0]); gB=np.log(B[:,0]/B[0,0])
lo,hi=max(gA[0],gB[0]),min(gA[-1],gB[-1]); mA=(gA>=lo)&(gA<=hi); mB=(gB>=lo)&(gB<=hi)
iL=np.exp(np.interp(gA[mA],gB[mB],np.log(B[mB,1])))
ck("symmetry: Lambda ratio = 2^-1/2","EXACT",float(np.mean(iL/A[mA,1])),2**-0.5,2e-3)
iG=np.exp(np.interp(gA[mA],gB[mB],np.log(B[mB,2])))
ck("symmetry: |grad xi| ratio = 1","EXACT",float(np.mean(iG/A[mA,2])),1.0,2e-3)
gam=[json.loads(l) for l in open("../runs/stream_F10CHK.jsonl") if l.strip()]
sg=[x["sup_gamma"] for x in gam if "sup_gamma" in x]
ck("transport invariant sup|r^2 u1|","EXACT",max(sg)/min(sg),1.0,1e-3)

# ---------- TIER C: fits (labeled; the attackable tier).
S={t:series(f"OR_z{t[0]}r{t[1]}", gated=True) for t in [(128,384),(128,768),(256,384),(256,768)]}
lo=max(v[0,0] for v in S.values()); hi=min(v[-1,0] for v in S.values())
sl={}
for k,v in S.items():
    m=(v[:,0]>=lo)&(v[:,0]<=hi)
    sl[k]=float(np.polyfit(np.log(v[m,0]),np.log(v[m,1]),1)[0])
ck("factorial slope Nz128 Nr384","FIT",sl[(128,384)],1.0159,2e-3)
ck("factorial slope Nz128 Nr768","FIT",sl[(128,768)],0.9873,2e-3)
ck("factorial slope Nz256 Nr384","FIT",sl[(256,384)],1.0027,2e-3)
ck("factorial slope Nz256 Nr768","FIT",sl[(256,768)],0.9978,2e-3)
ck("factorial spread","FIT",max(sl.values())-min(sl.values()),0.0286,2e-3)

print(f"{'claim':<34}{'tier':>6}{'recomputed':>14}{'reference':>12}{'PASS':>6}")
print("-"*74)
for n,k,v,r_,t,ok in R:
    print(f"{n:<34}{k:>6}{v:>14.6f}{r_:>12.6f}{'PASS' if ok else 'FAIL':>6}")
ex=[x for x in R if x[1]=="EXACT"]
print("-"*74)
print(f"totals: {sum(x[5] for x in R)}/{len(R)} pass | EXACT tier worst dev: "
      f"{max(abs(x[2]-x[3])/max(abs(x[3]),1e-12) for x in ex):.2e}")
for f in ["../runs/snap_SYMa/snap_SYMa_s1.h5","../runs/snap_SYMb/snap_SYMb_s1.h5"]:
    try:
        h=hashlib.md5(open(f,'rb').read()).hexdigest()[:12]
        print(f"md5 {f.split('/')[-1]}: {h}")
    except Exception: pass
