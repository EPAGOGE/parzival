import numpy as np, scipy.io as sio, os
base="/Users/epagogellc/parzival/refs/chen_hou/Perturbed_eqn/Computed profile"
for fn in ["temp600_991_10_15W_824_635.mat","Steady_state_pertb_oneMesh62036.mat"]:
    p=os.path.join(base,fn)
    d=sio.loadmat(p, struct_as_record=False, squeeze_me=True)
    print("="*70); print(fn)
    for k,v in d.items():
        if k.startswith("__"): continue
        try:
            print(f"  {k:16s} type={type(v).__name__} shape={getattr(v,'shape',None)}")
        except Exception as e: print(k,e)
    # scalars
    for k in ["cl","cw","r1","l0"]:
        if k in d: print("   ",k,"=",d[k])
