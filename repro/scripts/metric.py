import importlib.util, sys, numpy as np
sys.path.insert(0,".")
sp=importlib.util.spec_from_file_location("prg","polar_radial_gate.py")
prg=importlib.util.module_from_spec(sp); sys.modules["prg"]=prg; sp.loader.exec_module(prg)
alpha,gk,beta,g1,M = prg.alpha_and_gk()
mu=2.0+alpha
print("Re-scoring polar_radial_gate with a PER-POINT relative metric.")
print("The global L2 norm over 10.8 decades is dominated by the large-s end and")
print("structurally cannot see small-s error. That is the metric my gate used.\n")
print("  %3s %14s %14s %14s %14s" % ("k","RAW globalL2","RAW per-point","SUB globalL2","SUB per-point"))
wr_g=wr_p=ws_g=ws_p=0.0
for i,k in enumerate(range(1,7)):
    ck=gk[i]/((2.0*k)**2-mu**2)
    sg,A,_=prg.solve_raw(k,gk[i],alpha);   ex=ck*np.exp(mu*sg)
    rg=np.linalg.norm(A-ex)/np.linalg.norm(ex)
    rp=np.max(np.abs(A-ex)/np.maximum(np.abs(ex),1e-300))
    sg2,Pp,_=prg.solve_subst(k,gk[i],alpha); ex2=ck*np.ones_like(sg2)
    sgl=np.linalg.norm(Pp-ex2)/np.linalg.norm(ex2)
    spp=np.max(np.abs(Pp-ex2)/np.maximum(np.abs(ex2),1e-300))
    wr_g=max(wr_g,rg); wr_p=max(wr_p,rp); ws_g=max(ws_g,sgl); ws_p=max(ws_p,spp)
    print("  %3d %14.3e %14.3e %14.3e %14.3e" % (k,rg,rp,sgl,spp))
print("\n  worst RAW : globalL2 %.3e   per-point %.3e" % (wr_g,wr_p))
print("  worst SUB : globalL2 %.3e   per-point %.3e" % (ws_g,ws_p))
print("\n  global-L2 says SUBST is %.3gx better  -> the number my spec quoted (13x)" % (wr_g/max(ws_g,1e-300)))
print("  per-point says SUBST is %.3gx better  -> the number that matters" % (wr_p/max(ws_p,1e-300)))
# where does RAW's error live?
sg,A,_=prg.solve_raw(1,gk[0],alpha); ck=gk[0]/(4-mu**2); ex=ck*np.exp(mu*sg)
loc=np.abs(A-ex)/np.abs(ex)
o=np.argsort(sg)
print("\n  RAW k=1 per-point rel err vs s (the small-s end is where it lives):")
for j in range(0,len(o),len(o)//8):
    i=o[j]; print("     s=%6.2f  rel=%.3e" % (sg[i],loc[i]))
