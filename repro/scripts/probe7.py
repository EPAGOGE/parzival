import numpy as np, logging, sys
logging.getLogger('dedalus').setLevel(logging.ERROR)
sys.path.insert(0, "/Users/epagogellc/parzival/boussinesq")
import polar_tau2d_gate as G
from dedalus.libraries.matsolvers import matsolvers
print("registered matsolvers:", sorted(matsolvers))
print()
for name in sorted(matsolvers):
    if name in ("dummysolver","blockinverse","scipybanded","bandedqr","spqr_solve"): continue
    for Ns,Nb in [(32,24),(64,48)]:
        try:
            num, ex, tm, B = G.solve_generic_ms(Ns, Nb, name)
            g,_ = G.errors(num, ex)
            print(f" {name:36s} {Ns}x{Nb}  rel={g:.3e} max|tau|={tm:.3e}")
        except Exception as e:
            print(f" {name:36s} {Ns}x{Nb}  {type(e).__name__}: {str(e)[:55]}")
