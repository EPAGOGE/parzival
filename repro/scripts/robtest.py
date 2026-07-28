import sys, torch
sys.path.insert(0,"/Users/epagogellc/jlens")
from jlens import ModelTap, compute_jlens, Reader
from jlens.robustness import layer_conditioning, cell_fragility, eigen_conditioning, summarize

MODEL="HuggingFaceTB/SmolLM2-135M"; CACHE="/Users/epagogellc/jlens/cache"
corpus=[l.strip() for l in open("/Users/epagogellc/jlens/data/corpus_mini.txt") if l.strip()][:24]
tap=ModelTap(MODEL)
J=compute_jlens(tap,corpus,cache_dir=CACHE,progress=False)
print("J built for layers:",sorted(J))

print("\n=== LAYER CONDITIONING (singular values -- Lipschitz, always meaningful) ===")
lc=layer_conditioning(J)
print("  %6s %12s %12s %12s %10s"%("layer","sigma_max","sigma_min","cond","eff_rank"))
for l in sorted(lc):
    d=lc[l]; print("  %6d %12.4e %12.4e %12.4e %6d/%d"%(l,d["sigma_max"],d["sigma_min"],d["cond"],d["eff_rank"],d["dim"]))

prompt="The color of the planet fourth from the sun is"
print("\n=== CELL FRAGILITY: eps* = relative perturbation to J that flips the argmax ===")
cf=cell_fragility(tap,J,prompt)
pos=len(cf["tokens"])-1
print("  final position (token %r):"%cf["tokens"][pos])
print("  %6s %14s %12s %12s   %s"%("layer","argmax","gap","eps*","verdict"))
for l in cf["layers"]:
    e=cf["eps_star"][l][pos]; g=cf["gap"][l][pos]; a=cf["argmax"][l][pos]
    v="ROBUST" if e>=1e-1 else ("fragile" if e<1e-3 else "marginal")
    print("  %6d %14r %12.4g %12.4e   %s"%(l,a,g,e,v))
s=summarize(cf)
print("\n  overall across all cells: %s"%s["overall"])

print("\n=== EIGENVALUE DIAGNOSTIC (with the warning attached) ===")
ec=eigen_conditioning({l:J[l] for l in sorted(J)[:2]},top=4)
for l in sorted(ec):
    print("  layer %d  dep_normality=%.4f"%(l,ec[l]["dep_normality"]))
    for r in ec[l]["eigs"]:
        print("     lam=%+9.4f%+9.4fi  kappa=%10.3e  %s"
              %(r["re"],r["im"],r["kappa"],"trustworthy" if r["trustworthy"] else "NOT trustworthy"))
