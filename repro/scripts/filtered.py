import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)
M=pm.March(64,64,-2.0,25.0,filter_on=True)
I=(slice(2,-2),slice(2,-2)); O0=M.Ot0.copy(); sc=np.abs(O0[I]).max()
print("WITH Hou-Li filter (cutoff 0.65).  Unfiltered reference in parentheses.\n")
print("%8s %8s %11s %11s %11s %11s %10s" % ("step","tau","tail_s","tail_b","max|dOt|","drift","c_l"))
ref={0:(6.9e-7,7.1e-5,7.27e-2),1000:(4.7e-5,3.4e-3,1.14e-1),3000:(2.3e-4,1.6e-2,1.40e-1),6000:(2.3e-4,3.2e-2,1.94e-1)}
for k in range(12001):
    if k%1000==0:
        ts,tb=M.tails(M.Ot)
        dO,_,_,cl,cw,_=M.rhs(M.Ot,M.Bt)
        dr=np.abs(M.Ot[I]-O0[I]).max()/sc
        extra=""
        if k in ref: extra="   (unfilt: %.1e %.1e %.2e)"%ref[k]
        print("%8d %8.1f %11.4e %11.4e %11.4e %11.4e %10.6f%s"
              % (k,k*1e-3,ts,tb,np.abs(dO[I]).max(),dr,cl,extra),flush=True)
    M.step(1e-3)
print("\nfinal c_l=%.6f c_w=%.6f alpha=%.6f  (ref cl=%.6f cw=%.6f a=%.6f)"
      % (cl,cw,cw/cl,M.P["cl"],M.P["cw"],M.a0))
