import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)
print("RESOLUTION DISCRIMINATOR: does the drift SHRINK with N?")
print("  discretization artefact -> shrinks.  physical instability -> does not.\n")
for NS,NB in ((48,48),(64,64),(96,96)):
    M=pm.March(NS,NB,-2.0,25.0,filter_on=True)
    I=(slice(2,-2),slice(2,-2)); O0=M.Ot0.copy(); sc=np.abs(O0[I]).max()
    out=[]
    for k in range(6001):
        if k in (0,2000,4000,6000):
            dO,_,_,cl,_,_=M.rhs(M.Ot,M.Bt)
            out.append((k*1e-3,np.abs(dO[I]).max(),np.abs(M.Ot[I]-O0[I]).max()/sc,cl))
        M.step(1e-3)
    print("  %dx%d" % (NS,NB))
    for tau,res,dr,cl in out:
        print("     tau=%5.1f  max|dOt|=%10.4e  drift=%10.4e  c_l=%9.6f" % (tau,res,dr,cl),flush=True)
    print("     drift growth tau 2->6: %.3gx ;  c_l excursion: %.4f" %
          (out[-1][2]/max(out[1][2],1e-30), abs(out[-1][3]-M.P["cl"])),flush=True)
