"""Read every saved scan and produce the pseudospectral contour table + the verdict."""
import numpy as np, os
os.chdir("/private/tmp/claude-501/-Users-epagogellc/"
         "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
NL = {"A": 1.112468e+03, "B": 2.453998e+03}
print("="*92)
print("Q3  PSEUDOSPECTRAL PICTURE  --  where the resolvent is largest")
print("="*92)
for lab in ("A","B"):
    f=f"q345_imag_{lab}.npz"
    if not os.path.exists(f): continue
    d=np.load(f); ys,sy=d["ys"],d["sy"]
    print(f"\n[{lab}] sigma_min on the imaginary axis, ||L|| = {NL[lab]:.4e}")
    print(f"    {'y':>10s} {'sigma_min':>14s} {'||R(iy)||':>14s}")
    for y in (0,0.4,1,2,4,6,8,10,20,50,100,200,400,800):
        i=int(np.argmin(np.abs(ys-y)))
        print(f"    {ys[i]:10.2f} {sy[i]:14.6e} {1/sy[i]:14.6e}")
    k=int(np.argmin(sy))
    print(f"    argmin {ys[k]:.3f}i -> {sy[k]:.6e};  flat plateau: sigma_min stays within "
          f"{100*(sy[(ys<=6)].max()/sy[(ys<=6)].min()-1):.2f}% of its min over y in [0,6]")
for lab in ("A",):
    f=f"q345_win_{lab}.npz"
    if not os.path.exists(f): continue
    d=np.load(f); xs,yg,S=d["xs"],d["yg"],d["S"]
    print(f"\n[{lab}] SPEC WINDOW Re in [-2,2] x Im in [0,2] ({S.shape[1]}x{S.shape[0]}):"
          f" eps-contours, rightmost Re reached")
    print(f"    {'eps':>12s} {'rightmost Re of Lambda_eps':>28s} {'area frac':>11s}")
    for eps in (1e-11,1e-10,1e-9,1e-8,1e-6,1e-4,1e-3,3.9308e-3,3.94e-3,1e-2,3e-2,1e-1,3e-1,1.0):
        ins=S<=eps
        r = xs[np.argwhere(ins)[:,1]].max() if ins.any() else np.nan
        print(f"    {eps:12.4e} {r:>28.4f} {ins.mean():11.4f}"
              + ("   <-- eps*" if abs(eps-3.9308e-3)<1e-6 else ""))
    print(f"    min over the window = {S.min():.6e} at Re={xs[np.unravel_index(np.argmin(S),S.shape)[1]]:+.3f}"
          f" Im={yg[np.unravel_index(np.argmin(S),S.shape)[0]]:+.3f}  -> ||R|| = {1/S.min():.4e}")
    print(f"    sigma_min along Im=0 (the leftward fan):")
    j=0
    for x in (-2,-1.5,-1,-0.5,-0.25,0,0.25,0.5,1,2):
        i=int(np.argmin(np.abs(xs-x)))
        print(f"        Re={xs[i]:+6.2f}   sigma_min={S[j,i]:.6e}   ||R||={1/S[j,i]:.4e}")
for lab in ("A",):
    for nm,tag in (("q345_rhp_%s.npz","COARSE RHP h=20"),("q345_band_%s.npz","NEAR-AXIS BAND h=0.5")):
        f=nm%lab
        if not os.path.exists(f): continue
        d=np.load(f); xs,yg,S=d["xs"],d["yg"],d["S"]
        print(f"\n[{lab}] {tag}  {S.shape[1]}x{S.shape[0]}  Re in [{xs[0]:.0f},{xs[-1]:.0f}]"
              f" Im in [{yg[0]:.0f},{yg[-1]:.0f}]")
        print(f"    min overall = {S.min():.6e};  min on the Re=0 column = {S[:,0].min():.6e};"
              f"  min over Re>0 = {S[:,1:].min():.6e}")
        print(f"    -> {'AXIS carries the RHP minimum (no interior local max of ||R||)' if S[:,0].min()<=S[:,1:].min() else 'INTERIOR minimum -- theorem violated'}")
        K=0.0; zK=None
        for j in range(S.shape[0]):
            for i in range(S.shape[1]):
                if xs[i]>0 and xs[i]/S[j,i]>K: K=xs[i]/S[j,i]; zK=complex(xs[i],yg[j])
        print(f"    Kreiss from this grid: K >= {K:.6e} at z={zK}")
print("\n"+"="*92)
print("Q4  TRANSIENT GROWTH")
print("="*92)
for lab in ("A","B"):
    f=f"q345_real_{lab}.npz"
    if not os.path.exists(f): continue
    d=np.load(f); rr,sr=d["rr"],d["sr"]; kr=rr/sr; m=int(np.argmax(kr))
    print(f"[{lab}]  K >= {kr[m]:.6e} at z = {rr[m]:+.3f}  (sigma_min = {sr[m]:.6e})")
    print(f"      Re(z)||R(z)|| on the positive real axis:")
    for x in (0.1,0.25,0.5,1,2,4,8,16,32,64,128,256,512):
        i=int(np.argmin(np.abs(rr-x)))
        print(f"        z={rr[i]:8.2f}   ||R||={1/sr[i]:12.6e}   Re(z)||R||={kr[i]:12.6e}")
