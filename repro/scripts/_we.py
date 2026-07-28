import sys, os, numpy as np, time
os.chdir("/private/tmp/claude-501/-Users-epagogellc/d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
sys.path.insert(0,os.getcwd())
import q345, we_range
real,S,a,z = q345.load if False else q345.spectrum.load_production("A")
print("root alpha",a, "Nx",S.Nx,"Nb",S.Nb, flush=True)
xi = S.x
print("xi nodes:", xi.min(), xi.max(), "count", xi.size, flush=True)
# k -> infinity behaviour of max Re W(S(xi*,k)) at a spread of radial nodes
nodes = [1, 5, 10, 20, 30, 40, 50, 60, 68, 70]
print(f"{'node':>5s} {'xi':>9s} " + " ".join(f"{'k='+str(k):>12s}" for k in (10,100,1000,10000,100000)), flush=True)
for i in nodes:
    row=[]
    for k in (10,100,1000,10000,100000):
        hi,lo = we_range.max_re(S,z,i,k)
        row.append(hi)
    print(f"{i:5d} {xi[i]:9.4f} " + " ".join(f"{v:12.5e}" for v in row), flush=True)
