"""Q6 / GATE G3 -- repeat the decisive quantities at a second resolution AND a second
eps_b.  The campaign has already been burned once by treating an (N,XMAX) ladder as a
1-D cut of a 2-D surface, so eps_b is carried as an independent axis, not a constant."""
import importlib.util, pathlib, sys, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
H = pathlib.Path("/Users/epagogellc/parzival/boussinesq")
SCR = pathlib.Path("/private/tmp/claude-501/-Users-epagogellc/"
                   "d3b8cd8d-f2d4-467e-892a-c139848392b1/scratchpad")
def mod(n,f):
    s=importlib.util.spec_from_file_location(n,str(H/f)); m=importlib.util.module_from_spec(s)
    sys.modules[n]=m; s.loader.exec_module(m); return m
pc = mod("pc","polar_cornerreg.py"); pz = mod("pz","polar_zeros.py")

CASES = [("A  (16,40,12) eps_b=1e-4", (16,40,12), 1e-4, SCR/"hunt_fields/rung_00_a-0.344712.npz"),
         ("B  (24,56,12) eps_b=1e-4", (24,56,12), 1e-4, SCR/"q6_root_24_56_0.0001.npz"),
         ("C  (16,40,12) eps_b=5e-5", (16,40,12), 5e-5, SCR/"q6_root_16_40_5e-05.npz")]

for tag, degs, eps_b, path in CASES:
    d = np.load(path); a, zr = float(d["a"]), d["z"]
    S = pc.CornerRegSolver(edges=(0.0,2.0,15.0,25.0), degs=degs, Nb=36, eps_b=eps_b, alpha=a)
    S.adopt_seed(zr)
    Nx, Nb = S.Nx, S.Nb; n2 = Nx*Nb
    J = S.jacobian(zr).tocsr(); N = J.shape[0]
    liveT = np.setdiff1d(np.arange(n2), np.union1d(S.rT_pin, S.rT_c0))
    fr = np.concatenate([liveT, n2+liveT]); nf = fr.size
    part_l = np.array(sorted(S.rT_c0),dtype=int)
    part_r = np.array([S.partner[int(r)] for r in part_l],dtype=int)
    spp = np.concatenate([S.rP_bedge,S.rP_outer,S.rP_c0,S.rP_c1,S.rP_cornerI])
    coefP = np.broadcast_to(-(S.XI*S.G1**2),(Nx,Nb)).ravel().copy(); coefP[spp]=0.0
    lp = S._lp_factor(); Lp = J[2*n2:3*n2,2*n2:3*n2].tocsc()
    lpH = spla.splu(Lp.conj().T.tocsc())
    Jc=J.tocsc(); JH=J.conj().T.tocsr()
    Bc = np.asarray(J[:,[N-2,N-1]].todense())[fr,:]
    Cg = np.asarray(J[[N-2,N-1],:].todense())[:,fr]
    CB = Cg@Bc; Qc,_ = np.linalg.qr(Cg.T)
    res = S.residual(zr)
    print(f"\n{tag}   N={N} n_f={nf}  alpha={a:.8f}  h_id={S.h_id(zr):+.4e}  "
          f"||F||_rms={np.linalg.norm(res)/np.sqrt(res.size):.3e}", flush=True)
    def Zap(x):
        u=np.zeros(N,dtype=x.dtype); u[fr]=x
        u[part_l]=u[part_r]; u[n2+part_l]=u[n2+part_r]
        u[2*n2:3*n2]=lp.solve(coefP*u[:n2]); return u
    def ZapH(v):
        w=v[:2*n2].copy(); w[:n2]+=coefP.conj()*lpH.solve(v[2*n2:3*n2])
        t=w.copy(); t[part_r]+=w[part_l]; t[n2+part_r]+=w[n2+part_l]; return t[fr]
    def Mv(x): return (Jc@Zap(x))[fr]
    def MHv(w):
        v=np.zeros(N,dtype=w.dtype); v[fr]=w; return ZapH(JH@v)
    def Lv(x):
        y=Mv(x); return y-Bc@np.linalg.solve(CB,Cg@y)
    def LHv(y): return MHv(y-Cg.conj().T@np.linalg.solve(CB.conj().T,Bc.conj().T@y))
    Pk = lambda v: v-Qc@(Qc.T@v)
    # ||L||_2 by power iteration on the quotient
    v = Pk(np.random.default_rng(0).standard_normal(nf)); v/=np.linalg.norm(v); nrm=0.0
    for _ in range(300):
        y=Pk(Lv(v)); w=Pk(LHv(y)); nn=np.linalg.norm(w); v=w/nn; nrm=np.sqrt(nn)
    # omega(L) raw norm
    op = spla.LinearOperator((nf,nf), matvec=lambda v: Pk(0.5*(Lv(Pk(v))+LHv(Pk(v)))),
                             rmatvec=lambda v: Pk(0.5*(Lv(Pk(v))+LHv(Pk(v)))), dtype=float)
    om = float(spla.eigsh(op,k=1,which="LA",return_eigenvectors=False,tol=1e-8,maxiter=20000)[0])
    pn, smin = pz.proj_norm(Bc, Cg)
    # sigma_min(L) on ker(Cg) via 1/||R(0)||
    lu0 = spla.splu(Jc)
    def R0(f,h=False):
        r=np.zeros(N,dtype=float); r[fr]=f; return lu0.solve(r,trans=("H" if h else "N"))[fr]
    v=Pk(np.random.default_rng(1).standard_normal(nf)); v/=np.linalg.norm(v); s0=0.0
    for _ in range(120):
        y=Pk(R0(v)); w=Pk(R0(y,True)); nn=np.linalg.norm(w); v=w/nn; s0=np.sqrt(nn)
    print(f"    ||L||_2 = {nrm:.6e}   omega(L)_raw = {om:+.6e}   omega/||L|| = {om/nrm:.5f}",
          flush=True)
    print(f"    sigma_min(L|kerCg) = {1.0/s0:.6e}   ||R(0)|| = {s0:.6e}   "
          f"cond(L) = {nrm*s0:.4e}", flush=True)
    print(f"    ||P|| = {pn:.4f}  sin(theta_min) = {smin:.6e}   cond(Cg Bc) = "
          f"{np.linalg.cond(CB):.4f}", flush=True)
    cl, cw = float(zr[-2]), float(zr[-1])
    print(f"    far-field: c_l = {cl:.8f}   (c_w - a0 c_l) = {cw-a*cl:+.2e}   "
          f"=> max Re W_e = 0", flush=True)
    # a few resolvent points: the crossing level
    mask=np.zeros(N); mask[fr]=1.0; E=sp.diags(mask,format="csc")
    for zz in (0.0+1.0j, 0.0+2.0j, 1.0+0j, 2.0+0j):
        t0=time.time(); lu=spla.splu((zz*E-Jc).tocsc())
        def R(f,h=False):
            r=np.zeros(N,dtype=complex); r[fr]=f
            return lu.solve(r,trans=("H" if h else "N"))[fr]
        v=Pk(np.random.default_rng(2).standard_normal(nf).astype(complex)); v/=np.linalg.norm(v)
        s=0.0
        for _ in range(60):
            y=Pk(R(v)); w=Pk(R(y,True)); nn=np.linalg.norm(w); v=w/nn; s=np.sqrt(nn)
        print(f"      z={zz.real:+.1f}{zz.imag:+.1f}i  ||R||={s:.5e}  eps_cross=1/||R||="
              f"{1/s:.5e}  rel={1/s/nrm:.3e}  [{time.time()-t0:.0f}s]", flush=True)
