import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)
print("NODE DISTRIBUTION vs kte (N=64 on s in [-2,25]).")
print("target: more nodes in the TRANSITION band s in [2,15] where the residual lives.\n")
print("  %6s %9s %9s %11s   %s" % ("kte","min ds","max ds","ratio","nodes in [2,15]"))
for k in (0.0,0.90,0.99,0.999,0.9999):
    s,Ds,_=pm.grid(63,-2.0,25.0,kte=k)
    d=np.diff(s); n=((s>=2)&(s<=15)).sum()
    print("  %6.4g %9.4f %9.4f %11.1f   %d of 64" % (k,d.min(),d.max(),d.max()/d.min(),n))
