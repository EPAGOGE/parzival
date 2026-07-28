import numpy as np, logging, probe2
logging.getLogger('dedalus').setLevel(logging.ERROR)

def diag(Ns, Nb, variant="A"):
    solver, P, exact, taus = probe2.build(Ns, Nb, variant)
    solver.build_matrices(solver.subproblems, ['L'])
    sp = solver.subproblems[0]
    M = sp.L_min.toarray()
    U,S,Vt = np.linalg.svd(M)
    print(f"{variant} Ns={Ns} Nb={Nb} shape={M.shape}")
    print("  smallest 8 singular values / smax:", (S[-8:]/S[0]))
    # variable layout after pre_right compression
    vars_ = solver.problem.LHS_variables
    sizes = [sp.field_size(v) for v in vars_]
    names = [v.name for v in vars_]
    print("  vars:", list(zip(names,sizes)), "sum", sum(sizes))
    # pre_right maps compressed -> full var vector
    PR = sp.pre_right.toarray()      # (Jfull, Jcomp)
    nnull = int((S/S[0] < 1e-12).sum())
    print("  numerical nullity (sv/smax<1e-12):", nnull)
    if nnull:
        Ncomp = Vt[-nnull:].T                     # (Jcomp, nnull)
        Nfull = PR @ Ncomp                        # (Jfull, nnull)
        off = 0
        for nm, sz in zip(names, sizes):
            blk = Nfull[off:off+sz]
            print(f"    null energy in {nm:4s}: {np.linalg.norm(blk):.4e}  (max |comp| {np.abs(blk).max():.3e})")
            if nm != 'P' and np.linalg.norm(blk) > 1e-8:
                # which coefficient modes
                amp = np.abs(blk).max(axis=1)
                idx = np.where(amp > 1e-8*max(amp.max(),1e-300))[0]
                print(f"      active modes in {nm}: {idx.tolist()[:12]}  (field_size {sz})")
            off += sz
        # also: LEFT null space -> which equations are redundant
        Lnull = U[:, -nnull:]
        PL = sp.pre_left.toarray()   # (Icomp, Ifull)
        Lfull = PL.T @ Lnull
        eoff = 0
        for i, eqn in enumerate(solver.problem.equations):
            esz = sp.field_size(eqn['eqn'])
            blk = Lfull[eoff:eoff+esz]
            print(f"    LEFT null energy in eq{i}: {np.linalg.norm(blk):.4e}  (size {esz})")
            eoff += esz
    return M, S

if __name__ == "__main__":
    for v in ["A","deep"]:
        diag(16, 12, v); print()
    diag(24, 16, "A")
