import numpy as np
import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

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
    v: vector in T_pM (1,n+1)
    out: (1,n+1)
    """
    p=p/torch.sqrt(-ip(p,p)) # reprojects p onto the manifold, for precision
    theta=mag(v)
    if theta==0:
        out=p
    else:
        unitv=v/theta
        out=torch.cosh(theta)*p+torch.sinh(theta)*unitv
    out=out/torch.sqrt(-ip(out,out)) # reprojects out onto the manifold, for precision
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

def direct(p,xi):
    """
    p: point in H^n (1,n+1)
    xi: point in S^(n-1), the boundary at infinite (1,n)
    out: unit vector in T_pM in direction of xi (1,n+1)
    """
    p=p/torch.sqrt(-ip(p,p)) # reprojects p onto the manifold, for precision
    xi=torch.concat((torch.tensor([[1]]),xi),dim=1)
    out=xi+ip(p,xi)*p
    out=out/mag(out)
    return out

def loss(p, x, beta, xi):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    beta: number in [0,1)
    xi: point in S^(n-1), the boundary at infinite (1,n)
    """
    lpx=log(p,x)
    dpx=mag(lpx)
    out=torch.mean(dpx+ip(beta*direct(p,xi),lpx))
    return out

def grad(p, x, beta, xi):
    """
    p: point in H^n (1,n+1)
    x: batch of points in H^n (b,n+1)
    beta: number in [0,1)
    xi: point in S^(n-1), the boundary at infinite (1,n)
    out: gradient (1,n+1)
    """
    lpx=log(p,x)
    dpx=mag(lpx)
    xip=direct(p,xi)
    unitpx=lpx/torch.unsqueeze(dpx,1)
    cothdpx=torch.cosh(dpx)/torch.sinh(dpx)
    out=-unitpx-beta*(torch.unsqueeze((1-dpx*cothdpx)*ip(xip,unitpx)+dpx,1)*unitpx+torch.unsqueeze(dpx*(cothdpx-ip(xip,unitpx)),1)*xip)
    for j in [i for i, x in enumerate(dpx<1e-20) if x]: # just set to 0 when p and x_j are very close
        out[j,:]=-beta*xip[0,:]
    out=torch.mean(out,dim=0,keepdim=True)
    return out

def quantile(x, beta, xi, tol=1e-100):
    """
    x: batch of points in H^n (b,n+1)
    beta: number in [0,1)
    xi: point in S^(n-1), the boundary at infinite (1,n)
    out: (beta, xi)th-quantile (1,n+1)
    """
    x=x/torch.unsqueeze(torch.sqrt(-ip(x,x)),1) # reprojects x onto the manifold, for precision
    xi=xi/torch.sqrt(torch.sum(xi*xi)) # ensures xi is on S^n
    current_p=torch.unsqueeze(torch.concat((torch.ones(1),torch.zeros(x.shape[1]-1))),0) # initial estimate for quantile
    old_p=current_p.detach().clone()
    current_loss=loss(current_p,x,beta,xi)
    lr=0.001
    step=-grad(current_p,x,beta,xi)
    step/=mag(step)
    count=0
    while lr>tol and count<1000:
        new_p=exp(current_p,lr*step).float()
        new_loss=loss(new_p,x,beta,xi)
        #print(new_loss)
        if (new_loss<=current_loss):
            old_p=current_p
            current_p=new_p
            current_loss=new_loss
            step=-grad(current_p,x,beta,xi)
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
