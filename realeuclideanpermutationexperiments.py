import numpy as np
import math
import stat
import time
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, t

def check(u,X):
    n=X.shape[0]
    N=X.shape[1]
    M=u.shape[0]
    degenerates=-np.ones(M)
    for i in range(N):
        v=np.expand_dims(X[:,i],1)
        index=np.where(np.sum(X==v,axis=0)!=n)[0]
        diff=X[:,index]-v
        norm=np.sqrt(np.sum(diff**2,axis=0))
        Delta=np.expand_dims(np.sum(diff/np.expand_dims(norm,0), axis=1),0)+len(index)*u
        s1=np.sqrt(np.sum(Delta**2,axis=1))
        s2=(N-len(index))*(1+np.sqrt(np.sum(u**2,axis=1)))
        degenerates[np.where(s1<=s2)]=i
    return degenerates

def euclideanloss(uprime,Xprime,q):
    qprime=np.expand_dims(q,2)
    diff=Xprime-qprime
    norm=np.sqrt(np.sum(diff**2,axis=1)) # (M,N)
    inner=np.sum(diff*uprime,axis=1)
    return np.sum(norm,axis=1)+np.sum(inner,axis=1) # shape is (M,)

def euclideanquantile(u,X,tol=1e-10,most=30): # Newton's method
    X=np.transpose(X,(1,0))
    n=X.shape[0]
    N=X.shape[1]
    M=u.shape[0]
    out=np.zeros((M,n))
    checkers=check(u,X)
    deg1=(checkers!=-1)
    if np.any(deg1):
        out[deg1,:]=np.transpose(X[:,checkers[deg1].astype(int)])
    if np.any(~deg1):
        init_q=np.tile(np.expand_dims(np.median(X,axis=1),0),(M,1)) # (M,n)
        #init_q=np.tile(np.expand_dims(X[:,2],0),(M,1)) # (M,n)
        current_q=init_q.copy()
        Xprime=np.expand_dims(X,0) # (1,n,N)
        uprime=np.expand_dims(u,2) # (M,n,1)
        count=0
        step=np.zeros((M,n))
        deg2=np.repeat(False,M) # indices of u's that lead to singular Phis
        deg3=np.repeat(False,M) # indices of u's that have sufficiently converged
        deg=(deg1 | deg2) | deg3
        while (np.any(np.sum(step[~deg,:]**2,1)>tol) or count==0) and (count<most and np.sum(deg)<M): # Newton-Raphson
            current_q[~deg,:]+=step[~deg,:]
            count+=1
            current_qprime=np.expand_dims(current_q,2)
            diff=Xprime-current_qprime # (M,n,N)
            norm=np.sqrt(np.sum(diff**2,axis=1)) # (M,N)
            Delta=diff/np.expand_dims(norm,1)
            pos=np.where(norm==0)
            Delta[pos[0],:,pos[1]]=0
            Delta=np.sum(Delta,axis=2)+N*u
            norminv=1/norm
            norminv[norm==0]=0
            Phi=np.expand_dims(np.sum(norminv,axis=1),(1,2))*np.expand_dims(np.identity(n),0)-(diff*np.expand_dims(norminv**3,1))@np.transpose(diff,(0,2,1))
            if np.any((abs(np.linalg.det(Phi))<1e-10) & ~deg):
                deg2=(deg2 | ((abs(np.linalg.det(Phi))<1e-10) & ~deg))
                deg=(deg1 | deg2) | deg3
            step[~deg,:]=np.squeeze(np.linalg.inv(Phi[~deg,:,:])@np.expand_dims(Delta[~deg,:],2))
            if np.any((np.sum(step**2,1)<=tol) & ~deg):
                deg3=(deg3 | ((np.sum(step**2,1)<=tol) & ~deg))
                out[deg3,:]=current_q[deg3,:]
                deg=(deg1 | deg2) | deg3
        newdeg=(deg2 | ~deg) # indices of u's that have failed to sufficiently converge using Newton-Raphson. We'll use gradient descent instead
        if np.sum(newdeg>0): #gradient descent
            index=np.where(newdeg)[0]
            current_q=init_q.copy()
            current_qprime=np.expand_dims(current_q,2)
            new_q=current_q.copy()
            current_loss=euclideanloss(uprime[newdeg,:,:],Xprime,current_q[newdeg,:])
            lr=np.repeat(0.001,M)
            diff=Xprime-current_qprime[newdeg,:,:] # (M',n,N)
            norm=np.sqrt(np.sum(diff**2,axis=1)) # (M',N)
            step=np.sum(diff/np.expand_dims(norm,1),axis=2) # (M',n)
            pos=np.where(norm==0)
            step[pos[0],:]=0
            step+=N*u[newdeg,:]
            step/=np.sqrt(np.sum(step**2,axis=1,keepdims=True))
            count=0
            while np.any(lr[newdeg]>tol) and count<1000:
                count+=1
                new_q[newdeg,:]=current_q[newdeg,:]+np.expand_dims(lr[newdeg],1)*step
                new_loss=euclideanloss(uprime[newdeg,:,:],Xprime,new_q[newdeg,:])
                lr[newdeg]*=1.1*(new_loss<current_loss)+0.5*(new_loss>=current_loss) # adjust learning rate based on whether or not loss was improved
                current_q[index[new_loss<current_loss],:]=new_q[index[new_loss<current_loss],:].copy()
                current_qprime=np.expand_dims(current_q,2)
                diff[new_loss<current_loss,:,:]=Xprime-current_qprime[index[new_loss<current_loss],:,:]
                norm[new_loss<current_loss,:]=np.sqrt(np.sum(diff[new_loss<current_loss,:,:]**2,axis=1))
                step[new_loss<current_loss,:]=np.sum(diff[new_loss<current_loss,:,:]/np.expand_dims(norm[new_loss<current_loss,:],1),axis=2)
                pos=np.where(norm[new_loss<current_loss,:]==0)
                step[new_loss<current_loss,:][pos[0],:]=0
                step[new_loss<current_loss,:]+=N*u[index[new_loss<current_loss],:]
                step[new_loss<current_loss,:]/=np.sqrt(np.sum(step[new_loss<current_loss,:]**2,axis=1,keepdims=True))
                current_loss[new_loss<current_loss]=new_loss[new_loss<current_loss].copy()
            out[newdeg,:]=current_q[newdeg,:]
    return out

n=2
m=4
betas=np.array([0.5,0.9])
u=np.array([[0,0]])
for i in range(len(betas)):
    for j in range(m):
        u=np.concatenate((u,betas[i]*np.array([[np.cos(j*2*math.pi/(m)),np.sin(j*2*math.pi/(m))]])),axis=0)

data=np.load('Downloads/moignard_poincare_embedding.npz')
x=np.concatenate((data['x_train'],data['x_test']),axis=0)
y=np.concatenate((data['y_train'],data['y_test']),axis=0)

origin=torch.zeros(1,3)
origin[0,0]=1
x=torch.Tensor(x)
x=B2H(x)
x=log(origin,x)[:,1:3].numpy()

quantpvalues=np.array([])

distr1=x[y==2] # HF stage
distr2=x[y==3] # NP stage
#distr1=x[y==3] # NP stage
#distr2=x[y==4] # PS stage

insts=1000
draws=120
tolerance=1e-6
for inst in range(insts):
    np.random.seed(inst)
    x1=distr1[np.random.choice(distr1.shape[0],draws),:]
    x2=distr2[np.random.choice(distr2.shape[0],draws),:]
    x1quantile=euclideanquantile(u,x1,tol=1e-10,most=30)
    x2quantile=euclideanquantile(u,x2,tol=1e-10,most=30)
    quantstat=np.sum(np.sqrt(np.sum((x1quantile-x2quantile)**2,axis=1)))
    quantsum=0
    totalx=np.concatenate((x1,x2),0)
    reps=1000
    for k in range(reps):
        permindices=np.random.choice(totalx.shape[0],x1.shape[0],replace=False)
        permx1=totalx[permindices,:]
        permx2=np.delete(totalx,permindices,axis=0)
        permx1quantile=euclideanquantile(u,permx1,tol=1e-10,most=30)
        permx2quantile=euclideanquantile(u,permx2,tol=1e-10,most=30)
        permquantstat=np.sum(np.sqrt(np.sum((permx1quantile-permx2quantile)**2,axis=1)))
        quantsum+=(permquantstat>=quantstat)
        print(inst,k,quantsum)
    print(np.sum(quantpvalues>0.1),np.sum(quantpvalues>0.05),np.sum(quantpvalues>0.01),np.sum(quantpvalues>0.005),np.sum(quantpvalues>0.001),np.mean(quantpvalues),np.quantile(quantpvalues,np.array([0.25,0.5,0.75])))

print(np.mean(quantpvalues>0.1))
print(np.mean(quantpvalues>0.05))
print(np.mean(quantpvalues>0.01))
print(np.mean(quantpvalues>0.005))
print(np.mean(quantpvalues>0.001))

print(np.mean(quantpvalues))

