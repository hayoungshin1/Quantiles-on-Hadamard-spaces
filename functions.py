import numpy as np
import torch
from torchvision.utils import make_grid
import math
import matplotlib.pyplot as plt
from itertools import combinations
from operator import itemgetter

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
    old_p=current_p.detach().clone()
    current_loss=loss(current_p,x,beta,y,xiy)
    lr=0.001
    step=-grad(current_p,x,beta,y,xiy)
    step/=mag(step)
    count=0
    while lr>tol and count<1000:
        new_p=exp(current_p,lr*step).float()
        new_loss=loss(new_p,x,beta,y,xiy)
        if (new_loss<=current_loss):
            old_p=current_p
            current_p=new_p
            current_loss=new_loss
            step=-grad(current_p,x,beta,y,xiy)
            step/=mag(step)
            lr=1.1*lr # try to speed up convergence by increasing learning rate
            #count+=1
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
    #for j in [i for i, x in enumerate(t<1e-5) if x]: # log should be 0 when p=x_j
    #    out[j,:]=0
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
