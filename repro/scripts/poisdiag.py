import importlib.util, sys, numpy as np, numpy.linalg as la
sys.path.insert(0,"/Users/epagogellc/parzival/boussinesq")
def M(n,f):
    sp=importlib.util.spec_from_file_location(n,"/Users/epagogellc/parzival/boussinesq/"+f)
    m=importlib.util.module_from_spec(sp); sys.modules[n]=m; sp.loader.exec_module(m); return m
pc=M("pc","polar_corner.py")
print("IS THE POISSON OPERATOR LOSING ACCURACY WITH N?")
print("  Its d_xixi coefficient is g^2 ~ xi^2 ~ N^-4 at the first node, while Db2 is")
print("  O(N^2) -- so the operator becomes extremely anisotropic near the corner.")
print("  Test: solve with a MANUFACTURED rhs and measure the true residual.\n")
print("  %4s %10s %12s %12s %12s"%("N","1/g(node1)","||A x - b||/||b||","max|x|","est cond"))
rng=np.random.default_rng(0)
for N in (28,36,44,48,52,56,64):
    C=pc.Corner(N,N,25.0)
    n=C.nx*C.nb
    b=rng.standard_normal(n)
    b[C.brows]=0.0
    x=C.lu.solve(b)
    # reconstruct A to measure the true residual
    import scipy.sparse as sp
    g,mu=C.g,C.mu
    As=np.diag(g**2)@(C.Dx2+2*mu*C.Dx+mu**2*np.eye(C.nx))+np.diag(g*(1-g))@(C.Dx+mu*np.eye(C.nx))
    A=sp.kron(sp.csr_matrix(As),sp.identity(C.nb,format="csr"))+ \
      sp.kron(sp.identity(C.nx,format="csr"),sp.csr_matrix(C.Db2))
    A=sp.lil_matrix(A)
    rid=lambda i,j: i*C.nb+j
    for i in range(C.nx):
        for j in (0,C.nb-1):
            r=rid(i,j); A.rows[r],A.data[r]=[r],[1.0]
    for j in range(1,C.nb-1):
        r=rid(0,j); A.rows[r],A.data[r]=[r],[1.0]
        r=rid(C.nx-1,j); A.rows[r]=[rid(k,j) for k in range(C.nx)]; A.data[r]=list(C.Dx[C.nx-1,:])
    A=sp.csr_matrix(A)
    res=la.norm(A@x-b)/max(la.norm(b),1e-300)
    print("  %4d %10.1f %12.3e %12.3e %12s"
          %(N,1/C.g[1],res,np.abs(x).max(),"-"),flush=True)
print("\n  a residual growing rapidly with N means the prefactored LU is losing the")
print("  solve, and every Newton step inherits that error.")
