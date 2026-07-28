import importlib.util, sys, time, json
import numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)
NS,NB,DT,NSTEP = 64,64,1e-3,12000
M=pm.March(NS,NB,-2.0,25.0)
I=(slice(2,-2),slice(2,-2))
O0=M.Ot0.copy(); scale=np.abs(O0[I]).max()
print("grid %dx%d dt=%g steps=%d  ref cl=%.6f cw=%.6f" % (NS,NB,DT,NSTEP,M.P["cl"],M.P["cw"]),flush=True)
print("%8s %10s %12s %12s %11s %11s %11s" % ("step","tau","c_l","c_w","alpha","max|dOt|","drift"),flush=True)
t0=time.time(); hist=[]
for k in range(NSTEP):
    cl,cw,cond,dO,dB=M.step(DT)
    if k%400==0 or k==NSTEP-1:
        res=np.abs(dO[I]).max(); dr=np.abs(M.Ot[I]-O0[I]).max()/scale
        hist.append((k,k*DT,cl,cw,cw/cl,float(res),float(dr)))
        print("%8d %10.3f %12.7f %12.7f %11.7f %11.4e %11.4e"
              % (k,k*DT,cl,cw,cw/cl,res,dr),flush=True)
        if not np.isfinite(res) or res>1e3:
            print("DIVERGED",flush=True); break
print("wall %.1f min" % ((time.time()-t0)/60),flush=True)
json.dump(hist,open("/Users/epagogellc/parzival/runs/polar_march_long.json","w"))
