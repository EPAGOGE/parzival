import sympy as sp

s, b = sp.symbols('s beta', real=True)
cl, al = sp.symbols('c_l alpha', real=True, positive=False)
cw = al*cl

G = sp.Function('G')(b)   # Om angular
T = sp.Function('T')(b)   # B  angular
P = sp.Function('P')(b)   # Psi angular

Om  = sp.exp(al*s)*G
B   = sp.exp((1+2*al)*s)*T
Psi = sp.exp((2+al)*s)*P

def ds(f): return sp.diff(f, s)
def db(f): return sp.diff(f, b)

brk = lambda f: sp.exp(-2*s)*(ds(Psi)*db(f) - db(Psi)*ds(f))

E1 = cl*ds(Om) + brk(Om) - cw*Om - sp.exp(-s)*(sp.cos(b)*ds(B) - sp.sin(b)*db(B))
E2 = cl*ds(B)  + brk(B)  - (cl + 2*cw)*B
E3 = sp.exp(-2*s)*(sp.diff(Psi,s,2) + sp.diff(Psi,b,2)) + Om

for name, E in [('E1',E1), ('E2',E2), ('E3',E3)]:
    Es = sp.simplify(sp.expand(E))
    # collect powers of exp(s)
    poly = sp.collect(sp.expand(Es), sp.exp(s), evaluate=False)
    print('===', name)
    print(sp.simplify(Es))
    print('--- factored:')
    print(sp.factor_terms(sp.simplify(Es)))
