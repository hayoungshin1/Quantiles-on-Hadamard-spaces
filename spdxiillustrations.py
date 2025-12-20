import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

def ip(pinv, v1, v2):
    """
    pinv: one or a batch of p^{-1}, directly inputed to save time (1,m,m), or (b,m,m)
    v1, v2: batches of tangent vectors in T_pM (b,m,m)
    out: inner products (b)
    """
    t1=torch.matmul(pinv,v1)
    t2=torch.matmul(pinv,v2)
    out=torch.matmul(t1,t2).diagonal(offset=0,dim1=-1,dim2=-2).sum(-1)
    return out

def mag(pinv, v1):
    """
    pinv: one or a batch of p^{-1}, directly inputed to save time (1,m,m), or (b,m,m)
    v1: batch of tangent vectors in T_pM (b,m,m)
    out: size of each v1 (b)
    """
    sq=ip(pinv, v1, v1)
    out=torch.sqrt(torch.clamp(sq,min=0)) # ensures out is real
    return out

def matrix_exp(A):
    """
    A: batch of symmetric positive definite matrices (b,m,m)
    out: matrix exponentials of those matrices (b,m,m)
    """
    A=(A+torch.transpose(A,-2,-1))/2
    L, V=torch.linalg.eig(A)
    L=torch.real(L) # ensure realness
    V=torch.real(V) # ensure realness
    out=torch.matmul(torch.matmul(V,torch.diag_embed(torch.exp(L))),torch.linalg.inv(V))
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def matrix_log(A):
    """
    A: batch of symmetric positive definite matrices (b,m,m)
    out: matrix logs of those matrices (b,m,m)
    """
    A=(A+torch.transpose(A,-2,-1))/2
    L, V=torch.linalg.eig(A)
    L=torch.real(L) # ensure realness
    L=torch.clamp(L,min=0) # ensure non-negativeness
    V=torch.real(V) # ensure realness
    out=torch.matmul(torch.matmul(V,torch.diag_embed(torch.log(L))),torch.linalg.inv(V))
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def matrix_sqrt(A):
    """
    A: batch of symmetric positive definite matrices (b,m,m)
    out: matrix square roots of those matrices (b,m,m)
    """
    A=(A+torch.transpose(A,-2,-1))/2
    L, V=torch.linalg.eig(A)
    L=torch.real(L) # ensure realness
    L=torch.clamp(L,min=0) # ensure non-negativeness
    V=torch.real(V) # ensure realness
    out=torch.matmul(torch.matmul(V,torch.diag_embed(torch.sqrt(L))),torch.linalg.inv(V))
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def exp(phalf, phalfinv, v):
    """
    phalf: p^{1/2}, directly inputed to save time (1,m,m)
    phalfinv: p^{-1/2}, directly inputed to save time (1,m,m)
    v: batch of tangent vectors in each T_pM (b,m,m)
    out: each exp_p(v) (b,m,m)
    """
    v=(v+torch.transpose(v,-2,-1))/2
    out=matrix_exp(torch.matmul(torch.matmul(phalfinv,v),phalfinv))
    out=torch.matmul(torch.matmul(phalf,out),phalf)
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def log(xhalf, xhalfinv, p):
    """
    xhalf: batch of x^{1/2}, directly inputed to save time (b,m,m)
    xhalfinv: batch of x^{-1/2}, directly inputed to save time (b,m,m)
    p: point in P_n (1,m,m)
    out: each log_x(p) (b,m,m)
    """
    p=(p+torch.transpose(p,-2,-1))/2
    out=matrix_log(torch.matmul(torch.matmul(xhalfinv,p),xhalfinv))
    out=torch.matmul(torch.matmul(xhalf,out),xhalf)
    out=(out+torch.transpose(out,-2,-1))/2
    return out

def direct(xhalf, xhalfinv, yhalf, yhalfinv, xiy):
    """
    xhalf: batch of x^{1/2}, directly inputed to save time (b,m,m)
    xhalfinv: batch of x^{-1/2}, directly inputed to save time (b,m,m)
    yhalf: y^{1/2}, directly inputed to save time, where y is a point in M (1,m,m)
    yhalfinv: y^{-1/2}, directly inputed to save time, where y is a point in M (1,m,m)
    xiy: unit vector in T_yM corresponding to xi (1,m,m)
    out: each xi_x (b,m,m)
    """
    prod=torch.matmul(torch.matmul(yhalfinv,xiy),yhalfinv)
    D, V=torch.linalg.eig(prod)
    D=torch.real(D)
    V=torch.real(V)
    evals,indices=torch.sort(D,descending=True)
    evecs=torch.zeros_like(V)
    for i in range(D.shape[0]):
        evecs[i,:,:]=V[i,:,indices[i,:]]
    W=torch.matmul(torch.matmul(xhalfinv,yhalf),evecs)
    W[:,:,0]/=torch.squeeze(torch.sqrt(torch.matmul(torch.transpose(W[:,:,0:1],-2,-1),W[:,:,0:1])),2)
    for k in range(1,W.shape[-1]):
        W[:,:,k]-=torch.squeeze(torch.matmul(W[:,:,0:k],torch.matmul(torch.transpose(W[:,:,0:k],1,2),W[:,:,k:k+1])),2)
        W[:,:,k]/=torch.squeeze(torch.sqrt(torch.matmul(torch.transpose(W[:,:,k:k+1],-2,-1),W[:,:,k:k+1])),2)
    out=torch.matmul(torch.matmul(torch.matmul(torch.matmul(xhalf,W),torch.diag_embed(evals)),torch.transpose(W,1,2)),xhalf)
    out=(out+torch.transpose(out,-2,-1))/2
    return out



import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.data import get_fnames
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')
data, affine = load_nifti(hardi_fname)
from dipy.data import get_sphere
from dipy.viz import window, actor

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)
from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2)
tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(maskdata)

from dipy.reconst.dti import fractional_anisotropy, color_fa

FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0
FA = np.clip(FA, 0, 1)
RGB = color_fa(FA, tenfit.evecs)

area=slice(13,43), slice(44,74), slice(28,29)

evals = tenfit.evals[area]
evecs = tenfit.evecs[area]

cfa = RGB[area]
biggest=cfa.max()


full=tenfit.quadratic_form
full=torch.from_numpy(full)
x=full[area]
x=torch.flatten(x,0,2)
x=x[(100,200,300,400,500),:,:]
x=(x+torch.transpose(x,1,2))/2
x=x.float()
xinv=torch.linalg.inv(x)
xhalf=matrix_sqrt(x)
xhalfinv=torch.linalg.inv(xhalf)
identity=2*torch.tensor([[[1,0,0],[0,1,0],[0,0,1]]])/2

# visualizations

betas=torch.tensor([0,1,2,3,4])/10
finalevals=np.zeros((x.shape[0],len(betas),1,3))
finalevecs=np.zeros((x.shape[0],len(betas),1,3,3))
for i in range(3):
    finalevecs[:,:,:,i,i]=1

finalcfa=np.zeros((x.shape[0],len(betas),1,3))

for k in range(3):
    if k==0:
        xi1=torch.tensor([[[0,0,1],[0,0,0],[1,0,0]]])/np.sqrt(2)
    elif k==1:
        xi1=torch.tensor([[[0,0,0],[0,0,1],[0,1,0]]])/np.sqrt(2)
    elif k==2:
        xi1=torch.tensor([[[1,0,0],[0,1,0],[0,0,0]]])/np.sqrt(2)  
    xis=direct(xhalf, xhalfinv, identity, identity, xi1)
    for i in range(x.shape[0]):
        for j in range(len(betas)):
            ellipsoid=exp(torch.unsqueeze(xhalf[i],0), torch.unsqueeze(xhalfinv[i],0), torch.unsqueeze(j*xis[i],0))
            L, V=torch.linalg.eig(ellipsoid)
            L=torch.real(L)
            L=torch.clamp(L,min=0)
            V=torch.real(V)
            qevals,indices=torch.sort(L,descending=True)
            qevecs=torch.zeros_like(V)
            for l in range(L.shape[0]):
                qevecs[l,:,:]=V[l,:,indices[l,:]]
            qevals=torch.unsqueeze(torch.unsqueeze(qevals,1),1)
            qevecs=torch.unsqueeze(torch.unsqueeze(qevecs,1),1)
            qevals=qevals.numpy()
            qevecs=qevecs.numpy()
            qFA = fractional_anisotropy(qevals)
            qFA[np.isnan(qFA)] = 0
            qFA = np.clip(qFA, 0, 1)
            qRGB = color_fa(qFA, qevecs)
            qcfa=qRGB
            qcfa /= cfa.max()
            finalevals[j,i,0,:]=qevals[0,0,0,:]
            finalevecs[j,i,0,:,:]=qevecs[0,0,0,:,:]
            finalcfa[j,i,0,:]=qcfa[0,0,0,:]
    sphere = get_sphere('repulsion724')
    scene = window.Scene()
    scene.add(actor.tensor_slicer(finalevals, finalevecs, scalar_colors=finalcfa, sphere=sphere, scale=0.3, norm=True))
    scene.background((255,255,255))
    window.record(scene, n_frames=1, out_path='Documents/xis'+str(k)+'.png', size=(2000, 2000))
    scene.clear()
