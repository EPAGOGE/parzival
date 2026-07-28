import numpy as np
A=np.load("q345_ev_A.npy"); B=np.load("q345_ev_B.npy")
print(f"A: {A.size} eigenvalues  max Re {A.real.max():+.8f}  min Re {A.real.min():+.4f}  max|Im| {np.abs(A.imag).max():.3f}")
print(f"B: {B.size} eigenvalues  max Re {B.real.max():+.8f}  min Re {B.real.min():+.4f}  max|Im| {np.abs(B.imag).max():.3f}")
print()
# for each A eigenvalue, distance to the nearest B eigenvalue
d = np.array([np.abs(B-a).min() for a in A])
print("Q6 SPECTRAL CONVERGENCE: |lambda_A - nearest lambda_B|, binned by |lambda_A|")
print(f"{'|lambda| bin':>18s} {'count':>7s} {'median dist':>13s} {'best':>11s} {'worst':>11s}")
edges=[0,1,2,4,8,16,32,64,128,1e9]
for lo,hi in zip(edges[:-1],edges[1:]):
    m=(np.abs(A)>=lo)&(np.abs(A)<hi)
    if m.sum():
        print(f"  [{lo:6.0f},{hi:7.0f}) {m.sum():7d} {np.median(d[m]):13.4e} {d[m].min():11.3e} {d[m].max():11.3e}")
print()
print("The 24 rightmost A eigenvalues vs their nearest B partner:")
o=np.argsort(-A.real)[:24]
for i in o:
    j=int(np.argmin(np.abs(B-A[i])))
    print(f"   A {A[i].real:+.8f}{A[i].imag:+.8f}i   ->  B {B[j].real:+.8f}{B[j].imag:+.8f}i    dist {abs(A[i]-B[j]):.4e}")
print()
print("The 8 rightmost B eigenvalues vs their nearest A partner:")
for j in np.argsort(-B.real)[:8]:
    i=int(np.argmin(np.abs(A-B[j])))
    print(f"   B {B[j].real:+.8f}{B[j].imag:+.8f}i   ->  A {A[i].real:+.8f}{A[i].imag:+.8f}i    dist {abs(A[i]-B[j]):.4e}")
