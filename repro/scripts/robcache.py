import sys, torch, glob
sys.path.insert(0,"/Users/epagogellc/jlens")
# import the module directly -- avoids jlens/__init__ pulling in transformers
import importlib.util
sp=importlib.util.spec_from_file_location("rob","/Users/epagogellc/jlens/jlens/robustness.py")
rob=importlib.util.module_from_spec(sp); sys.modules["rob"]=rob; sp.loader.exec_module(rob)

f=sorted(glob.glob("/Users/epagogellc/jlens/cache/jlens_*.pt"))[0]
blob=torch.load(f, weights_only=False)
J={int(k):v for k,v in blob["J"].items()}
print("cache: %s"%f.split("/")[-1])
print("model: %s  layers: %s  corpus_size: %s\n"%(blob.get("model"),sorted(J),blob.get("corpus_size")))

print("=== LAYER CONDITIONING (singular values -- Lipschitz, always meaningful) ===")
lc=rob.layer_conditioning(J)
print("  %6s %12s %12s %12s %12s"%("layer","sigma_max","sigma_min","cond","eff_rank"))
for l in sorted(lc):
    d=lc[l]
    print("  %6d %12.4e %12.4e %12.4e %8d/%d"%(l,d["sigma_max"],d["sigma_min"],d["cond"],d["eff_rank"],d["dim"]))

print("\n=== EIGENVALUE DIAGNOSTIC -- is the eigen picture even usable here? ===")
ec=rob.eigen_conditioning(J,top=4)
for l in sorted(ec):
    print("  layer %2d  dep_normality = %.4f   (0 = normal; large => use singular values)"
          %(l,ec[l]["dep_normality"]))
    for r in ec[l]["eigs"]:
        print("      lam = %+10.4f %+10.4fi   kappa = %10.3e   %s"
              %(r["re"],r["im"],r["kappa"],"OK" if r["trustworthy"] else "NOT TRUSTWORTHY"))
