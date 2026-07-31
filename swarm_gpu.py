"""SWARM ENGINE — GPU version (JAX). Run: pip install -U "jax[cuda12]" ; python swarm_gpu.py
Same architecture validated on CPU 2026-07-22:
  batched dissipative-CLM  w_t = w*Hw - nu*Lambda w  as pure matmul+pointwise,
  slot recycling on fate resolution, bandit mass -> fate-boundary cells,
  confidence ledger with spectral-tail trust flags.
Scale knobs: B (batch), N (grid), MACRO (steps). On a 4090-class card start B=262144.
De Gregorio swap: replace rhs() with  u = W@Ginv (u_x = Hw)  advection form.
"""
import jax, jax.numpy as jnp, numpy as np, time
from functools import partial

N, B, NU = 128, 262_144, 1.0
key = jax.random.PRNGKey(3)
x  = 2*jnp.pi*jnp.arange(N)/N
k  = np.fft.rfftfreq(N, 1.0/N)
I  = np.eye(N)
def _H(c): wh=np.fft.rfft(c); wh*=-1j*np.sign(k); wh[0]=0; return np.fft.irfft(wh,n=N)
def _L(c): return np.fft.irfft(np.abs(k)*np.fft.rfft(c),n=N)
Hm = jnp.asarray(np.stack([_H(I[:,j]) for j in range(N)],1), jnp.float32)
Lm = jnp.asarray(np.stack([_L(I[:,j]) for j in range(N)],1), jnp.float32)

@jax.jit
def rhs(W): return W*(W@Hm.T) - NU*(W@Lm.T)

@jax.jit
def macro_step(W, t):
    M  = jnp.max(jnp.abs(W),axis=1)
    dt = jnp.minimum(2e-3, 0.08/jnp.maximum(1.0,M))[:,None]
    k1=rhs(W);k2=rhs(W+.5*dt*k1);k3=rhs(W+.5*dt*k2);k4=rhs(W+dt*k3)
    return W+(dt/6)*(k1+2*k2+2*k3+k4), t+dt[:,0]

# ---- host-side ledger / bandit (cheap; device does the flops) ----
NC=80; edges=np.linspace(3.0,8.0,NC+1); centers=.5*(edges[:-1]+edges[1:])
n=np.zeros(NC); blow=np.zeros(NC); lowtrust=np.zeros(NC); wts=np.ones(NC)/NC

def sample_A(m, rng):
    cells=rng.choice(NC,size=m,p=wts)
    return centers[cells]+(edges[1]-edges[0])*(rng.random(m)-0.5), cells

rng=np.random.default_rng(0)
A0,cell=sample_A(B,rng)
W=jnp.asarray(A0[:,None]*np.cos(np.asarray(x))[None,:],jnp.float32)
t=jnp.zeros(B); A0=jnp.asarray(A0)
t0=time.time(); steps=0
for it in range(4000):
    W,t=macro_step(W,t); steps+=B
    M=np.asarray(jnp.max(jnp.abs(W),axis=1))
    tl=np.asarray(t)
    done=np.where((M>1e3)|((M<0.1*np.asarray(A0))&(tl>0.5)))[0]
    if len(done):
        Wd=np.asarray(W[done])
        sp=np.abs(np.fft.rfft(Wd,axis=1)); frac=(sp[:,3*sp.shape[1]//4:]**2).sum(1)/np.maximum((sp**2).sum(1),1e-30)
        for j,ii in enumerate(done):
            c=cell[ii]; n[c]+=1; blow[c]+= M[ii]>1e3; lowtrust[c]+= frac[j]>1e-4
        Anew,cnew=sample_A(len(done),rng)
        W=W.at[done].set(jnp.asarray(Anew[:,None]*np.cos(np.asarray(x))[None,:],jnp.float32))
        A0=A0.at[done].set(jnp.asarray(Anew)); t=t.at[done].set(0.0)
        cell[done]=cnew
    if it%300==299:
        nn=np.maximum(n,1); p=blow/nn; se=np.sqrt(p*(1-p)/nn)
        b=((p>0.02)&(p<0.98)).astype(float)
        w2=0.10/NC + b*(se+0.02); wts[:]=w2/w2.sum()
        print(f"it={it+1} resolved={int(n.sum())} sps={steps/(time.time()-t0):,.0f}")
nn=np.maximum(n,1); p=blow/nn
print("A* boundary cells:", centers[(p>0.02)&(p<0.98)])
