import numpy as np
def Lagrag(x,a,b):
    x=np.atleast_1d(np.asarray(x,float)); a=np.asarray(a,float); b=np.asarray(b,float)
    F=np.zeros_like(x)
    for i in range(len(a)):
        m=np.full_like(x,b[i])
        for j in range(len(a)):
            if j!=i: m=m*(x-a[j])/(a[i]-a[j])
        F=F+m
    return F
def densera_exp(x,a1,a2,ref):
    x=np.asarray(x,float).copy()
    n=len(x); L=18; ad2=1
    i0=int(np.argmax(x>=a2))+1                 # MATLAB 1-based
    kk=3
    i1=int(np.argmax(x>=a1))+1-kk-L
    with np.errstate(divide='ignore'):
        xg=np.log(x[1:])-np.log(x[:-1])        # xg(k)=xg[k-1]
    g=lambda k: xg[np.asarray(k)-1]
    h0=g(i0)/ref
    fitx=np.concatenate([np.arange(i1,i1+kk+1), i1+L+kk+np.arange(0,kk+1)])
    fity=np.concatenate([g(np.arange(i1,i1+kk+1)), h0*np.ones(kk+1)])
    ls=np.arange(1,L+1)
    st=i1+kk
    mxg=Lagrag(i1+kk+ls,fitx,fity)
    ed0=x[i0-1]*np.exp(ad2*h0)
    i2=i0+5
    ad1=int(np.ceil((np.log(ed0/x[st])-mxg.sum())/h0))   # x(st+1) -> x[st]
    mxg=np.concatenate([mxg,np.ones(max(ad1,0))*h0])
    fitx2=np.concatenate([i2-L+np.arange(-kk,1), i2+np.arange(0,kk+1)])
    fity2=np.concatenate([h0*np.ones(kk+1), g(i2+np.arange(0,kk+1))])
    mxg=np.concatenate([mxg,Lagrag(i2-L+ls,fitx2,fity2)])
    mxg=np.concatenate([mxg,xg[i2:]])          # xg(i2+1:end) -> xg[i2:]
    return np.concatenate([x[:st+1], x[st]*np.exp(np.cumsum(mxg))]), ad1
def expmesh():
    n=355; N1=58; N2=38; Nexp1=90
    r0=1.025; r1=1.15; r3=2.4; Nexp=221; d=2
    N=n+Nexp; x=np.zeros(N); h=1/(N-1)
    h1=1/256.; h2=2/256.
    kk=4
    gd=np.concatenate([np.arange(-kk,1), np.arange(-kk,1)+N2])
    gh=np.concatenate([h1*np.ones(kk+1), h2*np.ones(kk+1)])
    I1=np.arange(1,N1+1); x[I1-1]=(I1-1)*h1
    dx=Lagrag(np.arange(1,N2+1),gd,gh)
    for i in range(1,N2+1): x[N1+i-1]=x[N1+i-2]+dx[i-1]
    N3=n-N1-N2; h3=1/N3; yv=np.arange(1,N3+1)*h3
    M=h2/h3
    r=np.log(r0/(1+h3))/((1+h3)**d-1)
    x[N1+N2:n]=M*yv*np.exp(r*yv**d)+x[N1+N2-1]
    ad=np.ones(Nexp); rat=1.0
    for i in range(1,Nexp+1):
        lam=min((i-1)/Nexp1,1.0)
        rat=rat*np.exp((1-lam)*np.log(r0)+lam*np.log(r1)); ad[i-1]=rat
    x[n:n+Nexp]=x[n-1]*ad
    print("  pre-Densera: N=%d  x[57]=%.6f x[95]=%.6f x[354]=%.6f x[-1]=%.6e"%(len(x),x[57],x[95],x[354],x[-1]))
    x,a1=densera_exp(x,30,40000,2.1); print("  after Densera(30,4e4,2.1): N=%d ad1=%d x[-1]=%.6e"%(len(x),a1,x[-1]))
    x,a2=densera_exp(x,30,3000,r3);   print("  after Densera(30,3e3,2.4): N=%d ad1=%d x[-1]=%.6e"%(len(x),a2,x[-1]))
    dxdy=np.zeros(len(x)); dxdy[1:]=(x[1:]-x[:-1])/h; dxdy[0]=h1/h
    return x,dxdy,h
x,dxdy,h=expmesh()
print("\nlength(x1) = %d   (code names the save file with this -> 'Steady_state_pertb%d_Nlevcor4')"%(len(x),len(x)))
print("x1[-1]  = %.6e   s=ln = %.4f"%(x[-1],np.log(x[-1])))
print("r_max = sqrt2*x1[-1] = %.6e  s=%.4f"%(np.sqrt(2)*x[-1],np.log(np.sqrt(2)*x[-1])))
print("last ratios:", (x[-1]/x[-2], x[-2]/x[-3]))
print("r_st = x1(end-60) = %.6e (s=%.3f)"%(x[-61],np.log(x[-61])))
print("r_ed = x1(end-12) = %.6e (s=%.3f)"%(x[-13],np.log(x[-13])))
ex=28; r1=1.15
gx_end=x[-1]*r1**ex
print("gx1(end) after Meshext(ex=28, ratio 1.15) = %.6e  s=%.3f"%(gx_end,np.log(gx_end)))
N1e=len(x)+ex-5
print("esr max (N1=%d extended grid) approx = %.6e"%(N1e, np.sqrt(2)*gx_end))
