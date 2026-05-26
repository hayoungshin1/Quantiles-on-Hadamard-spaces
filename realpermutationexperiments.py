n=2
m=4
origin=torch.zeros(1,3)
origin[0,0]=1
originradial= torch.zeros(m,3)
angles=2*torch.arange(0,m)*math.pi/m
originradial[:,1]=torch.cos(angles)
originradial[:,2]=torch.sin(angles)
betas=torch.tensor([0.5,0.9])

data=np.load('Downloads/moignard_poincare_embedding.npz')
x=np.concatenate((data['x_train'],data['x_test']),axis=0)
y=np.concatenate((data['y_train'],data['y_test']),axis=0)
x=torch.Tensor(x)
colors=['tab:red', 'tab:purple', 'tab:green', 'tab:orange', 'tab:blue']

f = plt.figure(figsize=(7,7))
ax = plt.gca()
for color in range(5):
    distr1=x[y==color]
    distr1.shape
    plt.scatter(distr1.cpu().numpy()[:,0], distr1.cpu().numpy()[:,1], s=30, c=colors[color], marker = '.')
    circle = plt.Circle((0, 0), 1, color='b', fill=False)
    ax.add_patch(circle)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    plt.axis('equal')
    plt.show(block=False)

for color in range(3,5):
    distr1=x[y==color]
    distr1.shape
    f = plt.figure(figsize=(7,7))
    ax = plt.gca()
    plt.scatter(distr1.cpu().numpy()[:,0], distr1.cpu().numpy()[:,1], s=30, c=colors[color], marker = '.')
    circle = plt.Circle((0, 0), 1, color='b', fill=False)
    ax.add_patch(circle)
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    plt.axis('equal')
    plt.show(block=False)


x=B2H(x)

## permutation experiments

permindivquantpvalues=(1+len(betas)*len(originradial))*[np.array([])]
permbetaquantpvalues=len(betas)*[np.array([])]#####
permmeanpvalues=np.array([])
permquantpvalues=np.array([])

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
    x1mean=torch.from_numpy(np.squeeze(fastmean(np.expand_dims(x1,(0,3)),tol=tolerance),(0,3)))
    x2mean=torch.from_numpy(np.squeeze(fastmean(np.expand_dims(x2,(0,3)),tol=tolerance),(0,3)))
    if torch.isnan(x1mean).any().item() or torch.isnan(x2mean).any().item(): # fastmean sometimes gives nans
        x1mean=frechetmean(x1)
        x2mean=frechetmean(x2)
    meanstat=torch.sum(mag(newlog(x1mean,x2mean))).item()
    indivquantstats=np.zeros(1+len(betas)*len(originradial))
    x1quantile=quantile(x1, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
    x2quantile=quantile(x2, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
    indivquantstats[0]=torch.sum(mag(newlog(x1quantile,x2quantile))).item()
    betaquantstats=np.zeros(len(betas))
    for i in range(len(betas)):
        betaquantstats[i]=0
        for j in range(len(originradial)):
            x1quantile=quantile(x1, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)
            x2quantile=quantile(x2, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)
            indivquantstats[i*len(originradial)+j+1]=torch.sum(mag(newlog(x1quantile,x2quantile))).item()
            betaquantstats[i]+=indivquantstats[i*len(originradial)+j+1]
    quantstat=np.sum(indivquantstats)
    indivquantsums=(1+len(betas)*len(originradial))*[0]
    betaquantsums=len(betas)*[0]
    meansum=0
    quantsum=0
    totalx=torch.concat((x1,x2),0)
    reps=1000
    for k in range(reps):
        permindices=np.random.choice(totalx.shape[0],x1.shape[0],replace=False)
        permx1=totalx[permindices,:]
        permx2=np.delete(totalx,permindices,axis=0)
        permx1mean=torch.from_numpy(np.squeeze(fastmean(np.expand_dims(permx1,(0,3)),tol=tolerance),(0,3)))
        permx2mean=torch.from_numpy(np.squeeze(fastmean(np.expand_dims(permx2,(0,3)),tol=tolerance),(0,3)))
        if torch.isnan(permx1mean).any().item() or torch.isnan(permx2mean).any().item(): # fastmean sometimes gives nans
            permx1mean=frechetmean(permx1)
            permx2mean=frechetmean(permx2)
        permmeanstat=torch.sum(mag(newlog(permx1mean,permx2mean))).item()
        meansum+=(permmeanstat>=meanstat)
        permindivquantstats=np.zeros(1+len(betas)*len(originradial))
        permx1quantile=quantile(permx1, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
        permx2quantile=quantile(permx2, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
        permindivquantstats[0]=torch.sum(mag(newlog(permx1quantile,permx2quantile))).item()
        indivquantsums[0]+=(permindivquantstats[0]>=indivquantstats[0]).item()
        permbetaquantstats=np.zeros(len(betas))
        for i in range(len(betas)):
            permbetaquantstats[i]=0
            for j in range(len(originradial)):
                permx1quantile=quantile(permx1, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)
                permx2quantile=quantile(permx2, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)
                permindivquantstats[i*len(originradial)+j+1]=torch.sum(mag(newlog(permx1quantile,permx2quantile))).item()
                indivquantsums[i*len(originradial)+j+1]+=(permindivquantstats[i*len(originradial)+j+1]>=indivquantstats[i*len(originradial)+j+1]).item()
                permbetaquantstats[i]+=permindivquantstats[i*len(originradial)+j+1]
            betaquantsums[i]+=(permbetaquantstats[i]>=betaquantstats[i]).item()
        permquantstat=np.sum(permindivquantstats)
        quantsum+=(permquantstat>=quantstat)
        print(inst,k,indivquantsums,betaquantsums,meansum,quantsum)
    for i in range(len(permindivquantpvalues)):
        permindivquantpvalues[i]=np.append(permindivquantpvalues[i],indivquantsums[i]/reps)
    for i in range(len(permbetaquantpvalues)):
        permbetaquantpvalues[i]=np.append(permbetaquantpvalues[i],betaquantsums[i]/reps)
    permmeanpvalues=np.append(permmeanpvalues,meansum/reps)
    permquantpvalues=np.append(permquantpvalues,quantsum/reps)
    for i in range(len(permindivquantpvalues)):
        print(np.sum(permindivquantpvalues[i]>0.1),np.sum(permindivquantpvalues[i]>0.05),np.sum(permindivquantpvalues[i]>0.01),np.sum(permindivquantpvalues[i]>0.005),np.sum(permindivquantpvalues[i]>0.001),np.mean(permindivquantpvalues[i]),np.quantile(permindivquantpvalues[i],np.array([0.25,0.5,0.75])))
    for i in range(len(permbetaquantpvalues)):
        print(np.sum(permbetaquantpvalues[i]>0.1),np.sum(permbetaquantpvalues[i]>0.05),np.sum(permbetaquantpvalues[i]>0.01),np.sum(permbetaquantpvalues[i]>0.005),np.sum(permbetaquantpvalues[i]>0.001),np.mean(permbetaquantpvalues[i]),np.quantile(permbetaquantpvalues[i],np.array([0.25,0.5,0.75])))#####
    print(np.sum(permmeanpvalues>0.1),np.sum(permmeanpvalues>0.05),np.sum(permmeanpvalues>0.01),np.sum(permmeanpvalues>0.005),np.sum(permmeanpvalues>0.001),np.mean(permmeanpvalues),np.quantile(permmeanpvalues,np.array([0.25,0.5,0.75])))
    print(np.sum(permquantpvalues>0.1),np.sum(permquantpvalues>0.05),np.sum(permquantpvalues>0.01),np.sum(permquantpvalues>0.005),np.sum(permquantpvalues>0.001),np.mean(permquantpvalues),np.quantile(permquantpvalues,np.array([0.25,0.5,0.75])))

print(np.mean(permindivquantpvalues[0]>0.1),np.mean(permmeanpvalues>0.1),np.mean(permquantpvalues>0.1))
print(np.mean(permindivquantpvalues[0]>0.05),np.mean(permmeanpvalues>0.05),np.mean(permquantpvalues>0.05))
print(np.mean(permindivquantpvalues[0]>0.01),np.mean(permmeanpvalues>0.01),np.mean(permquantpvalues>0.01))
print(np.mean(permindivquantpvalues[0]>0.005),np.mean(permmeanpvalues>0.005),np.mean(permquantpvalues>0.005))
print(np.mean(permindivquantpvalues[0]>0.001),np.mean(permmeanpvalues>0.001),np.mean(permquantpvalues>0.001))
print(np.mean(permindivquantpvalues[0]>permquantpvalues),np.mean(permindivquantpvalues[0]<permquantpvalues))
print(np.mean(permmeanpvalues>permquantpvalues),np.mean(permmeanpvalues<permquantpvalues))

for i in range(len(permindivquantpvalues)):
    print(np.mean(permindivquantpvalues[i]))
for i in range(len(permbetaquantpvalues)):
    print(np.mean(permbetaquantpvalues[i]))
print(np.mean(permmeanpvalues),np.mean(permquantpvalues))

## asymptotic experiments

insts=1000
draws=120
tolerance=1e-6
K=len(betas)*m+1

asympindivquantpvalues=np.zeros((K,insts))
asympbetaquantpvalues=np.zeros((len(betas),insts))
asympmeanpvalues=np.zeros(insts)
asympquantpvalues=np.zeros(insts)

distr1=x[y==2] # HF stage
distr2=x[y==3] # NP stage
#distr1=x[y==3] # NP stage
#distr2=x[y==4] # PS stage

for inst in range(insts):
    np.random.seed(inst)
    x1=distr1[np.random.choice(distr1.shape[0],draws),:]
    x2=distr2[np.random.choice(distr2.shape[0],draws),:]
    x1mean=torch.from_numpy(np.squeeze(fastmean(np.expand_dims(x1,(0,3)),tol=tolerance)))[1:]
    x2mean=torch.from_numpy(np.squeeze(fastmean(np.expand_dims(x2,(0,3)),tol=tolerance)))[1:]
    if torch.isnan(x1mean).any().item() or torch.isnan(x2mean).any().item(): # fastmean sometimes gives nans
        x1mean=torch.squeeze(frechetmean(x1))[1:]
        x2mean=torch.squeeze(frechetmean(x2))[1:]
    x1quantiles=torch.squeeze(quantile(x1, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance))[1:]
    x2quantiles=torch.squeeze(quantile(x2, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance))[1:]
    for i in range(len(betas)):
        for j in range(m):
            x1quantiles=torch.concat((x1quantiles,torch.squeeze(quantile(x1, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance))[1:]),dim=0)
            x2quantiles=torch.concat((x2quantiles,torch.squeeze(quantile(x2, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance))[1:]),dim=0)
    x1meanauto=x1mean.clone().requires_grad_(True)
    x2meanauto=x2mean.clone().requires_grad_(True)
    x1meanjacobians=torch.zeros(n,draws)
    x1meanhessians=torch.zeros(draws,n,n)
    x2meanjacobians=torch.zeros(n,draws)
    x2meanhessians=torch.zeros(draws,n,n)
    for draw in range(draws):
        meanlossauto1 = lambda p_: meanloss(torch.cat([torch.sqrt(1+torch.sum(p_**2)).unsqueeze(0),p_]).unsqueeze(0),x1[draw:(draw+1),:])
        meanlossauto2 = lambda p_: meanloss(torch.cat([torch.sqrt(1+torch.sum(p_**2)).unsqueeze(0),p_]).unsqueeze(0),x2[draw:(draw+1),:])
        x1meanjacobians[:,draw]=jacobian(meanlossauto1,x1meanauto)
        x1meanhessians[draw,:,:]=hessian(meanlossauto1,x1meanauto)
        x2meanjacobians[:,draw]=jacobian(meanlossauto2,x2meanauto)
        x2meanhessians[draw,:,:]=hessian(meanlossauto2,x2meanauto)
    meanlambdainv1=torch.linalg.inv(torch.mean(x1meanhessians,dim=0))
    meansigma1=torch.cov(x1meanjacobians)
    meanvar1=torch.matmul(meanlambdainv1,torch.matmul(meansigma1,meanlambdainv1))
    meanlambdainv2=torch.linalg.inv(torch.mean(x2meanhessians,dim=0))
    meansigma2=torch.cov(x2meanjacobians)
    meanvar2=torch.matmul(meanlambdainv2,torch.matmul(meansigma2,meanlambdainv2))
    meanstat=draws*torch.matmul(x1mean-x2mean,torch.matmul(torch.linalg.inv(meanvar1+meanvar2),x1mean-x2mean)).item()
    asympmeanpvalues[inst]=1-chi2.cdf(meanstat,n)
    x1quantilesauto=x1quantiles.clone().requires_grad_(True)
    x2quantilesauto=x2quantiles.clone().requires_grad_(True)
    x1quantjacobians=torch.zeros(K*n,draws)
    x1quanthessians=torch.zeros(draws,K*n,K*n)
    x2quantjacobians=torch.zeros(K*n,draws)
    x2quanthessians=torch.zeros(draws,K*n,K*n)
    for draw in range(draws):
        quantilelossauto1 = lambda p_: loss(torch.cat([torch.sqrt(1+torch.sum(p_**2)).unsqueeze(0),p_]).unsqueeze(0),x1[draw:(draw+1),:], 0, origin, torch.unsqueeze(originradial[0,:],0))
        quantilelossauto2 = lambda p_: loss(torch.cat([torch.sqrt(1+torch.sum(p_**2)).unsqueeze(0),p_]).unsqueeze(0),x2[draw:(draw+1),:], 0, origin, torch.unsqueeze(originradial[0,:],0))
        x1quantjacobians[0:n,draw]=jacobian(quantilelossauto1,x1quantilesauto[0:n])
        x1quanthessians[draw,0:n,0:n]=hessian(quantilelossauto1,x1quantilesauto[0:n])
        x2quantjacobians[0:n,draw]=jacobian(quantilelossauto2,x2quantilesauto[0:n])
        x2quanthessians[draw,0:n,0:n]=hessian(quantilelossauto2,x2quantilesauto[0:n])
    for i in range(len(betas)):
        for j in range(m):
            for draw in range(draws):
                quantilelossauto1 = lambda p_: loss(torch.cat([torch.sqrt(1+torch.sum(p_**2)).unsqueeze(0),p_]).unsqueeze(0),x1[draw:(draw+1),:], betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0))
                quantilelossauto2 = lambda p_: loss(torch.cat([torch.sqrt(1+torch.sum(p_**2)).unsqueeze(0),p_]).unsqueeze(0),x2[draw:(draw+1),:], betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0))
                x1quantjacobians[(i*m+j+1)*n:(i*m+j+2)*n,draw]=jacobian(quantilelossauto1,x1quantilesauto[(i*m+j+1)*n:(i*m+j+2)*n])
                x1quanthessians[draw,(i*m+j+1)*n:(i*m+j+2)*n,(i*m+j+1)*n:(i*m+j+2)*n]=hessian(quantilelossauto1,x1quantilesauto[(i*m+j+1)*n:(i*m+j+2)*n])
                x2quantjacobians[(i*m+j+1)*n:(i*m+j+2)*n,draw]=jacobian(quantilelossauto2,x2quantilesauto[(i*m+j+1)*n:(i*m+j+2)*n])
                x2quanthessians[draw,(i*m+j+1)*n:(i*m+j+2)*n,(i*m+j+1)*n:(i*m+j+2)*n]=hessian(quantilelossauto2,x2quantilesauto[(i*m+j+1)*n:(i*m+j+2)*n])
    quantlambdainv1=torch.linalg.inv(torch.mean(x1quanthessians,dim=0))
    quantsigma1=torch.cov(x1quantjacobians)
    quantvar1=torch.matmul(quantlambdainv1,torch.matmul(quantsigma1,quantlambdainv1))
    quantlambdainv2=torch.linalg.inv(torch.mean(x2quanthessians,dim=0))
    quantsigma2=torch.cov(x2quantjacobians)
    quantvar2=torch.matmul(quantlambdainv2,torch.matmul(quantsigma2,quantlambdainv2))
    quantstat=draws*torch.matmul(x1quantiles-x2quantiles,torch.matmul(torch.linalg.inv(quantvar1+quantvar2),x1quantiles-x2quantiles)).item()
    asympquantpvalues[inst]=1-chi2.cdf(quantstat,K*n)
    for k in range(K):
        indivquantstat=draws*torch.matmul(x1quantiles[k*n:(k+1)*n]-x2quantiles[k*n:(k+1)*n],torch.matmul(torch.linalg.inv(quantvar1[k*n:(k+1)*n,k*n:(k+1)*n]+quantvar2[k*n:(k+1)*n,k*n:(k+1)*n]),x1quantiles[k*n:(k+1)*n]-x2quantiles[k*n:(k+1)*n])).item()
        asympindivquantpvalues[k,inst]=1-chi2.cdf(indivquantstat,n)
    for i in range(len(betas)):
        betaquantstat=draws*torch.matmul(x1quantiles[(i*m+1)*n:((i+1)*m+1)*n]-x2quantiles[(i*m+1)*n:((i+1)*m+1)*n],torch.matmul(torch.linalg.inv(quantvar1[(i*m+1)*n:((i+1)*m+1)*n,(i*m+1)*n:((i+1)*m+1)*n]+quantvar2[(i*m+1)*n:((i+1)*m+1)*n,(i*m+1)*n:((i+1)*m+1)*n]),x1quantiles[(i*m+1)*n:((i+1)*m+1)*n]-x2quantiles[(i*m+1)*n:((i+1)*m+1)*n])).item()
        asympbetaquantpvalues[i,inst]=1-chi2.cdf(betaquantstat,m*n)
    for k in range(K):
        print(np.sum(asympindivquantpvalues[k]>0.1),np.sum(asympindivquantpvalues[k]>0.05),np.sum(asympindivquantpvalues[k]>0.01),np.sum(asympindivquantpvalues[k]>0.005),np.sum(asympindivquantpvalues[k]>0.001),np.mean(asympindivquantpvalues[k]),np.quantile(asympindivquantpvalues[k],np.array([0.25,0.5,0.75])))
    for i in range(len(betas)):
        print(np.sum(asympbetaquantpvalues[i]>0.1),np.sum(asympbetaquantpvalues[i]>0.05),np.sum(asympbetaquantpvalues[i]>0.01),np.sum(asympbetaquantpvalues[i]>0.005),np.sum(asympbetaquantpvalues[i]>0.001),np.mean(asympbetaquantpvalues[i]),np.quantile(asympbetaquantpvalues[i],np.array([0.25,0.5,0.75])))#####
    print(np.sum(asympmeanpvalues>0.1),np.sum(asympmeanpvalues>0.05),np.sum(asympmeanpvalues>0.01),np.sum(asympmeanpvalues>0.005),np.sum(asympmeanpvalues>0.001),np.mean(asympmeanpvalues),np.quantile(asympmeanpvalues,np.array([0.25,0.5,0.75])))
    print(np.sum(asympquantpvalues>0.1),np.sum(asympquantpvalues>0.05),np.sum(asympquantpvalues>0.01),np.sum(asympquantpvalues>0.005),np.sum(asympquantpvalues>0.001),np.mean(asympquantpvalues),np.quantile(asympquantpvalues,np.array([0.25,0.5,0.75])))
    print(inst)

print(np.mean(asympindivquantpvalues[0]>0.1),np.mean(asympmeanpvalues>0.1),np.mean(asympquantpvalues>0.1))
print(np.mean(asympindivquantpvalues[0]>0.05),np.mean(asympmeanpvalues>0.05),np.mean(asympquantpvalues>0.05))
print(np.mean(asympindivquantpvalues[0]>0.01),np.mean(asympmeanpvalues>0.01),np.mean(asympquantpvalues>0.01))
print(np.mean(asympindivquantpvalues[0]>0.005),np.mean(asympmeanpvalues>0.005),np.mean(asympquantpvalues>0.005))
print(np.mean(asympindivquantpvalues[0]>0.001),np.mean(asympmeanpvalues>0.001),np.mean(asympquantpvalues>0.001))
print(np.mean(asympindivquantpvalues[0]>asympquantpvalues),np.mean(asympindivquantpvalues[0]<asympquantpvalues))
print(np.mean(asympmeanpvalues>asympquantpvalues),np.mean(asympmeanpvalues<asympquantpvalues))

for i in range(len(asympindivquantpvalues)):
    print(np.mean(asympindivquantpvalues[i]))
for i in range(len(asympbetaquantpvalues)):
    print(np.mean(asympbetaquantpvalues[i]))
print(np.mean(asympmeanpvalues),np.mean(asympquantpvalues))
