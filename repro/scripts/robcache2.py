import sys, torch, glob, importlib.util
sp=importlib.util.spec_from_file_location("rob","/Users/epagogellc/jlens/jlens/robustness.py")
rob=importlib.util.module_from_spec(sp); sys.modules["rob"]=rob; sp.loader.exec_module(rob)
f=sorted(glob.glob("/Users/epagogellc/jlens/cache/jlens_*.pt"))[0]
blob=torch.load(f, weights_only=False); J={int(k):v for k,v in blob["J"].items()}
print("model: %s   layers: %s\n"%(blob.get("model"),sorted(J)))
print("=== LAYER CONDITIONING (singular values: Lipschitz, always meaningful) ===")
lc=rob.layer_conditioning(J)
print("  %5s %12s %12s %12s %14s"%("layer","sigma_max","sigma_min","cond","eff_rank"))
for l in sorted(lc):
    d=lc[l]; print("  %5d %12.4e %12.4e %12.4e %8d / %d"
                   %(l,d["sigma_max"],d["sigma_min"],d["cond"],d["eff_rank"],d["dim"]))
print("\n=== EIGEN DIAGNOSTIC: real vs complex, after the degeneracy fix ===")
ec=rob.eigen_conditioning(J,top=4)
nreal_ok=ncplx_bad=0
for l in sorted(ec):
    for r in ec[l]["eigs"]:
        if abs(r["im"])<1e-9 and r["trustworthy"]: nreal_ok+=1
        if abs(r["im"])>=1e-9 and not r["trustworthy"]: ncplx_bad+=1
for l in (0, sorted(ec)[len(ec)//2], sorted(ec)[-1]):
    print("  layer %2d  dep_normality=%.4f"%(l,ec[l]["dep_normality"]))
    for r in ec[l]["eigs"]:
        print("     lam=%+9.4f%+9.4fi  kappa=%10.3e  %-16s %s"
              %(r["re"],r["im"],r["kappa"],
                "OK" if r["trustworthy"] else "NOT TRUSTWORTHY", r.get("note","")))
print("\n  across all layers, top-4 each: %d real eigenvalues TRUSTWORTHY, "
      "%d complex NOT trustworthy"%(nreal_ok,ncplx_bad))
