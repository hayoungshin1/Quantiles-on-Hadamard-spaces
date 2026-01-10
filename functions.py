import numpy as np
import torch
from torchvision.utils import make_grid
import math
import matplotlib.pyplot as plt
from itertools import combinations
from operator import itemgetter
from scipy.spatial.distance import cdist, euclidean
from sklearn.covariance import MinCovDet, EmpiricalCovariance

def ip(v1, v2):
    """
    v1: batch of vectors (b,n+1) or (1,n+1)
    v2: batch of vectors (b,n+1) or (1,n+1)
    out: (b)
    """
    copy=v1.detach().clone()
    copy[:,0]=-copy[:,0]
    out=torch.sum(copy*v2, dim=1)
    return out

def mag(v1):
    """
    v1: batch of vectors (b,n+1)
    out: (b)
    """
    sq=ip(v1, v1)
    out=torch.sqrt(torch.clamp(sq,min=0)) # ensures out is real
    return out

def exp(p, v):
    """
    p: point in H^n (1,n+1)
    v: batch of vectors in T_pH^n (b,n+1)
    out: (b,n+1)
    """
    p=p/torch.sqrt(-ip(p,p)) # reprojects p onto the manifold, for precision
    theta=torch.unsqueeze(mag(v),1)
    unitv=v/theta
    out=torch.cosh(theta)*p+torch.sinh(theta)*unitv
    for each in ((theta==0).nonzero())[:,0]:
        out[each,:]=torch.squeeze(p,0)
    out=out/torch.unsqueeze(torch.sqrt(-ip(out,out)),1) # reprojects out onto the manifold, for precision
    return out

def log(p, x):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    out: (b,n+1)
    """
    p=p/torch.sqrt(-ip(p,p)) # reprojects p onto the manifold, for precision
    a=ip(p,x)
    a=torch.clamp(a,max=-1) # ensures -a is at least 1
    theta=torch.acosh(-a)
    v=x+torch.matmul(torch.unsqueeze(ip(p,x),1),p)
    t=torch.unsqueeze(mag(v),1)
    unitv=v/t
    out=torch.unsqueeze(theta,1)*unitv
    return out

def alog(x, p):
    """
    x: batch of points in H^n (b,n+1)
    p: point in H^n (1,n+1)
    out: each log_x(p) (b,n+1)
    """
    p=p/torch.sqrt(-ip(p,p)) # reprojects p onto the manifold, for precision
    x=x/torch.unsqueeze(torch.sqrt(-ip(x,x)),dim=1) # reprojects x onto the manifold, for precision
    a=ip(p,x)
    a=torch.clamp(a,max=-1) # ensures -a is at least 1
    theta=torch.acosh(-a)
    v=p+torch.unsqueeze(ip(x,p),1)*x
    t=torch.unsqueeze(mag(v),1)
    unitv=v/t
    out=torch.unsqueeze(theta,1)*unitv
    return out

def newlog(y, z):
    """
    y: batch of points in H^n (b,n+1)
    z: point in H^n (b,n+1)
    out: each log_y(z) (b,n+1)
    """
    y=y/torch.unsqueeze(torch.sqrt(-ip(y,y)),dim=1) # reprojects y onto the manifold, for precision
    z=z/torch.unsqueeze(torch.sqrt(-ip(z,z)),dim=1) # reprojects z onto the manifold, for precision
    a=ip(y,z)
    a=torch.clamp(a,max=-1) # ensures -a is at least 1
    theta=torch.acosh(-a)
    v=z+torch.unsqueeze(ip(y,z),1)*y
    t=torch.unsqueeze(mag(v),1)
    unitv=v/t
    out=torch.unsqueeze(theta,1)*unitv
    return out

def direct(p,y,xiy):
    """
    p: point in H^n (1,n+1)
    y: point in H^n (1,n+1)
    xiy: unit vector in T_yH^n corresponding to xi (1,n)
    out: unit vector in T_pH^n corresponding to xi (1,n+1)
    """
    p=p/torch.sqrt(-ip(p,p)) # reprojects p onto the manifold, for precision
    y=y/torch.sqrt(-ip(y,y)) # reprojects y onto the manifold, for precision
    out=y+xiy+ip(p,y+xiy)*p
    out=out/mag(out)
    return out

def loss(p, x, beta, y, xiy):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    beta: number in [0,1)
    y: point in H^n (1,n+1)
    xiy: unit vector in T_yH^n corresponding to xi (1,n)
    out: loss
    """
    lpx=log(p,x)
    dpx=mag(lpx)
    out=torch.mean(dpx+ip(beta*direct(p,y,xiy),lpx))
    return out

def grad(p, x, beta, y, xiy):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    beta: number in [0,1)
    y: point in H^n (1,n+1)
    xiy: unit vector in T_yH^n corresponding to xi (1,n)
    out: gradient (1,n+1)
    """
    lpx=log(p,x)
    dpx=mag(lpx)
    xip=direct(p,y,xiy)
    unitpx=lpx/torch.unsqueeze(dpx,1)
    cothdpx=torch.cosh(dpx)/torch.sinh(dpx)
    out=-unitpx-beta*(torch.unsqueeze((1-dpx*cothdpx)*ip(xip,unitpx)+dpx,1)*unitpx+torch.unsqueeze(dpx*(cothdpx-ip(xip,unitpx)),1)*xip)
    for j in [i for i, x in enumerate(dpx<1e-20) if x]: # just set to 0 when p and x_j are very close
        out[j,:]=-beta*xip[0,:]
    out=torch.mean(out,dim=0,keepdim=True)
    return out

def quantile(x, beta, y, xiy, tol=1e-100):
    """
    x: batch of points in H^n (b,n+1)
    beta: number in [0,1)
    y: point in H^n (1,n+1)
    xiy: unit vector in T_yH^n corresponding to xi (1,n)
    out: (beta, xi)th-quantile (1,n+1)
    """
    x=x/torch.unsqueeze(torch.sqrt(-ip(x,x)),1) # reprojects x onto the manifold, for precision
    xiy=xiy/torch.sqrt(torch.sum(xiy*xiy)) # ensures xiy is unit vector
    current_p=torch.unsqueeze(torch.concat((torch.ones(1),torch.zeros(x.shape[1]-1))),0) # initial estimate for quantile
    current_loss=loss(current_p,x,beta,y,xiy)
    lr=0.001
    step=-grad(current_p,x,beta,y,xiy)
    step/=mag(step)
    count=0
    while lr>tol and count<1000:
        new_p=exp(current_p,lr*step).float()
        new_loss=loss(new_p,x,beta,y,xiy)
        if (new_loss<=current_loss):
            current_p=new_p
            current_loss=new_loss
            step=-grad(current_p,x,beta,y,xiy)
            step/=mag(step)
            lr=1.1*lr # try to speed up convergence by increasing learning rate
        else:
            lr=lr/2
            count+=1
    out=current_p
    return out

def H2B(p):
    """
    p: batch of points in H^n (b,n+1)
    out: p sent to the Poincare ball B^n (b,n)
    """
    out=p[:,1:]/torch.unsqueeze(p[:,0]+1,1)
    return out

def B2H(x):
    '''
    A map from B^n to H^n
    x:      torch.tensor whose size = (b, n)
    out:    torch.tensor whose size = (b, n + 1)
    '''
    norm_square = (torch.norm(x, dim = 1) ** 2).unsqueeze(dim=1)
    out = torch.cat([(1 + norm_square)/(1 - norm_square), 2 * x / (1 - norm_square)], dim = 1)
    return out

def meanloss(p, x):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    out: loss
    """
    lpx=log(p,x)
    dpx=mag(lpx)
    out=torch.mean(dpx**2)
    return out

def meangrad(p, x):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    out: gradient (1,n+1)
    """
    out=-log(p,x)
    out=torch.mean(out,dim=0,keepdim=True)
    return out

def frechetmean(x, tol=1e-100):
    """
    x: batch of points in H^n (b,n+1)
    out: frechet mean (1,n+1)
    """
    x=x/torch.unsqueeze(torch.sqrt(-ip(x,x)),1) # reprojects x onto the manifold, for precision
    current_p=torch.unsqueeze(torch.concat((torch.ones(1),torch.zeros(x.shape[1]-1))),0) # initial estimate for quantile
    old_p=current_p.detach().clone()
    current_loss=meanloss(current_p,x)
    lr=0.001
    step=-meangrad(current_p,x)
    step/=mag(step)
    count=0
    while lr>tol and count<1000:
        new_p=exp(current_p,lr*step).float()
        new_loss=meanloss(new_p,x)
        if (new_loss<=current_loss):
            old_p=current_p
            current_p=new_p
            current_loss=new_loss
            step=-meangrad(current_p,x)
            step/=mag(step)
            lr=1.1*lr # try to speed up convergence by increasing learning rate
        else:
            lr=lr/2
            count+=1
    out=current_p
    return out

def pt(x, v, p):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    v: batch of vectors in tangent spaces at x (b,n+1)
    out: v parallel transported to T_pH^n (b,n+1)
    """
    lxp=alog(x,p)
    dxp=mag(lxp)
    out=v-torch.unsqueeze(ip(lxp,v)/(dxp)**2,1)*(log(p,x)+lxp)
    return out

def vorth(v,orth):
    """
    v: batch of vectors in a tangent space (b,n+1)
    basis: an orthonormal basis in the same tangent space (n,n+1)
    out: v in terms of the basis (b,n)
    """
    copy=v.detach().clone()
    copy[:,0]=-copy[:,0]
    out=torch.matmul(copy,torch.transpose(orth,0,1))
    return out

def transform(p,basis,v,transformers):
    """
    p: point (1,n+1)
    basis: an orthonormal basis in T_pH^n (n,n+1)
    v: batch of vectors in T_pH^n in terms of basis (b,n)
    transformers: invertible matrix used to transform v (n,n)
    out: batch of transformed vectors, not in terms of basis (b,n+1)
    """
    out=torch.matmul(v,transformers)
    out=torch.matmul(out,basis)
    return out

def geometric_median(X, eps=1e-10):
    """
    Code obtained from https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points, by user orlp
    X: numpy array of data points
    """
    y = np.mean(X, 0)
    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]
        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)
        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y
        if euclidean(y, y1) < eps:
            return y1
        y = y1

def ip2(p,q):
    """
    p: B batches of N' points in hyperboloid (B,1 or N',n+1,1)
    q: B batches of N' points in hyperboloid (B,1 or N',n+1,1)
    out: B batches of inner products of p and q (B,1 or N',1,1)
    """
    newq=q.copy()
    newq[:,:,0,:]*=-1
    out=np.sum(p*newq,axis=2,keepdims=True)
    return out

def fastmean(Y,tol=1e-100):
    """
    Y: B batches of N data points in hyperboloid (B,N,n+1,1)
    out: B batches of weighted Fr'echet means  (B,1,n+1,1)
    """
    N=np.shape(Y)[1]
    old=Y[:,0,:,:]
    old=np.expand_dims(old,1) #(B,1,n+1,1)
    current=np.copy(old) #(B,1,n+1,1)
    count=1
    while (count==1 or np.sum(np.arccosh(-ip2(old,current))>tol)>0) and count<1000:
        prod=-ip2(Y,current) #(B,N,1,1)
        paran=2*np.arccosh(prod)/np.sqrt(prod**2-1) #(B,N,1,1)
        indices=np.where(np.abs(prod-1)<tol)[0:2]
        paran[indices]=2
        u=paran*Y/N #(B,N,n+1,1)
        usum=np.sum(u,axis=1,keepdims=True) #(B,1,n+1,1)
        denom=np.sqrt(np.absolute(-ip2(usum,usum)))
        old=np.copy(current)
        current=usum/denom #(B,1,n+1,1)
        count+=1
    out=current
    return out
