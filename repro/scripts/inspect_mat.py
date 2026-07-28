import scipy.io as sio, numpy as np, h5py, os
p1 = os.path.expanduser("~/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/temp600_991_10_15W_824_635.mat")
p2 = os.path.expanduser("~/parzival/refs/chen_hou/Perturbed_eqn/Computed profile/Steady_state_pertb_oneMesh62036.mat")
for p in (p1,p2):
    print("="*70); print(os.path.basename(p))
    try:
        d = sio.loadmat(p)
        for k,v in d.items():
            if k.startswith("__"): continue
            try: print(f"  {k}: shape={np.shape(v)} dtype={getattr(v,'dtype',None)}")
            except Exception as e: print("  ",k,e)
    except Exception as e:
        print("  sio failed:", e)
        with h5py.File(p,'r') as f:
            f.visit(lambda n: print("  ",n))
