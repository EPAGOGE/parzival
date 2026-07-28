import numpy as np, traceback
from scipy.optimize import least_squares
t=np.linspace(0.8,1.53,30); TM=t.max()
tau=0.73/np.log(0.25/0.015); A=0.25*np.exp(0.8/tau); d=A*np.exp(-t/tau)
def res(p):
    c,s,r=p
    m=np.exp(c)*(TM+np.exp(s)-t)**np.exp(r)
    return np.log(m)-np.log(d)
s0=np.log(0.18); r0=np.log(2.6); c0=np.log(d[0]/(TM+np.exp(s0)-t[0])**np.exp(r0))
print("x0=",[c0,s0,r0], "res0 finite?", np.all(np.isfinite(res([c0,s0,r0]))))
try:
    sol=least_squares(res,[c0,s0,r0],method='lm',maxfev=40000)
    print("OK cost",sol.cost, sol.x)
except Exception as e:
    traceback.print_exc()
