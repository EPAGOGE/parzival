import importlib.util, sys, numpy as np
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
sp=importlib.util.spec_from_file_location("pm","/Users/epagogellc/parzival/boussinesq/polar_march.py")
pm=importlib.util.module_from_spec(sp); sys.modules["pm"]=pm; sp.loader.exec_module(pm)
print("INNER BC, third option: Robin with the LOCALLY MEASURED slope from the seed,")
print("per-beta, instead of the asymptotic constant (which is 5% wrong at s=-2).\n")
M0=pm.March(64,64,-2.0,25.0,filter_on=True)
D0=M0.Ds[0]
qO=(D0 @ M0.Ot0)/np.where(np.abs(M0.Ot0[0])>1e-300,M0.Ot0[0],np.nan)
qB=(D0 @ M0.Bt0)/np.where(np.abs(M0.Bt0[0])>1e-300,M0.Bt0[0],np.nan)
print("  measured inner log-slope at s=-2:  Ot median %.4f  Bt median %.4f"
      % (np.nanmedian(qO), np.nanmedian(qB)))
print("  asymptotic (corner Taylor):        Ot %.4f  Bt %.4f  <- what the failed run used\n"
      % (1-M0.a0, 1-2*M0.a0))
def run(mode, qo=None, qb=None, n=6001):
    M=pm.March(64,64,-2.0,25.0,filter_on=True,inner=("dirichlet" if mode=="frozen" else "x"))
    if mode!="frozen":
        D0=M.Ds[0]
        def proj():
            M.Ot[0,:]=(D0[1:] @ M.Ot[1:,:])/(qo-D0[0])
            M.Bt[0,:]=(D0[1:] @ M.Bt[1:,:])/(qb-D0[0])
        M.robin_project=proj
    I=(slice(2,-2),slice(2,-2)); O0=M.Ot0.copy(); sc=np.abs(O0[I]).max()
    rows=[]
    for k in range(n):
        if k%2000==0:
            dO,_,_,cl,_,_=M.rhs(M.Ot,M.Bt)
            rows.append((k*1e-3,np.abs(dO[I]).max(),np.abs(M.Ot[I]-O0[I]).max()/sc,cl))
        M.step(1e-3)
    return rows
for lab,args in (("frozen Dirichlet",("frozen",None,None)),
                 ("Robin, LOCAL slope",("robin",np.nanmedian(qO),np.nanmedian(qB))),
                 ("Robin, asymptotic",("robin",1-M0.a0,1-2*M0.a0))):
    print("  --- %s ---" % lab)
    for tau,res,dr,cl in run(*args):
        print("     tau=%5.1f  res=%10.4e  drift=%10.4e  c_l=%9.6f" % (tau,res,dr,cl),flush=True)
