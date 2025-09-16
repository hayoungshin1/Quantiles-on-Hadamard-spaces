n=2
m=4
origin=torch.zeros(1,3)
origin[0,0]=1
originradial= torch.zeros(m,3)
angles=2*torch.arange(0,m)*math.pi/m
originradial[:,1]=torch.cos(angles)
originradial[:,2]=torch.sin(angles)
betas=torch.tensor([0.5,0.9])
#betas=torch.tensor([0.8])

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
indivquantpvalues=(1+len(betas)*len(originradial))*[np.array([])]
betaquantpvalues=len(betas)*[np.array([])]#####
meanpvalues=np.array([])
quantpvalues=np.array([])

distr1=x[y==3]
distr2=x[y==4]

insts=1000
draws=120
tolerance=1e-6
for inst in range(insts):
    np.random.seed(inst)
    x1=distr1[np.random.choice(distr1.shape[0],draws),:]
    x2=distr2[np.random.choice(distr2.shape[0],draws),:]
    x1mean=torch.from_numpy(np.squeeze(fastmean(np.expand_dims(x1,(0,3)),tol=tolerance),(0,3)))
    x2mean=torch.from_numpy(np.squeeze(fastmean(np.expand_dims(x2,(0,3)),tol=tolerance),(0,3)))
    meanstat=torch.sum(mag(newlog(x1mean,x2mean))).item()
    indivquantstats=np.zeros(1+len(betas)*len(originradial))
    x1quantile=quantile(x1, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
    x2quantile=quantile(x2, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
    indivquantstats[0]=torch.sum(mag(newlog(x1quantile,x2quantile))).item()
    betaquantstats=np.zeros(len(betas))#####
    for i in range(len(betas)):
        betaquantstats[i]=0#####
        for j in range(len(originradial)):
            x1quantile=quantile(x1, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)
            x2quantile=quantile(x2, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)
            indivquantstats[i*len(originradial)+j+1]=torch.sum(mag(newlog(x1quantile,x2quantile))).item()
            betaquantstats[i]+=indivquantstats[i*len(originradial)+j+1]#####
    quantstat=np.sum(indivquantstats)
    indivquantsums=(1+len(betas)*len(originradial))*[0]
    betaquantsums=len(betas)*[0]#####
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
        permmeanstat=torch.sum(mag(newlog(permx1mean,permx2mean))).item()
        meansum+=(permmeanstat>=meanstat)
        permindivquantstats=np.zeros(1+len(betas)*len(originradial))
        permx1quantile=quantile(permx1, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
        permx2quantile=quantile(permx2, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
        permindivquantstats[0]=torch.sum(mag(newlog(permx1quantile,permx2quantile))).item()
        indivquantsums[0]+=(permindivquantstats[0]>=indivquantstats[0]).item()
        permbetaquantstats=np.zeros(len(betas))#####
        for i in range(len(betas)):
            permbetaquantstats[i]=0#####
            for j in range(len(originradial)):
                permx1quantile=quantile(permx1, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)
                permx2quantile=quantile(permx2, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)
                permindivquantstats[i*len(originradial)+j+1]=torch.sum(mag(newlog(permx1quantile,permx2quantile))).item()
                indivquantsums[i*len(originradial)+j+1]+=(permindivquantstats[i*len(originradial)+j+1]>=indivquantstats[i*len(originradial)+j+1]).item()
                permbetaquantstats[i]+=permindivquantstats[i*len(originradial)+j+1]#####
            betaquantsums[i]+=(permbetaquantstats[i]>=betaquantstats[i]).item()#####
        permquantstat=np.sum(permindivquantstats)
        quantsum+=(permquantstat>=quantstat)
        print(inst,k,indivquantsums,betaquantsums,meansum,quantsum)
    for i in range(len(indivquantpvalues)):
        indivquantpvalues[i]=np.append(indivquantpvalues[i],indivquantsums[i]/reps)
    for i in range(len(betaquantpvalues)):#####
        betaquantpvalues[i]=np.append(betaquantpvalues[i],betaquantsums[i]/reps)#####
    meanpvalues=np.append(meanpvalues,meansum/reps)
    quantpvalues=np.append(quantpvalues,quantsum/reps)
    for i in range(len(indivquantpvalues)):
        print(np.sum(indivquantpvalues[i]>0.1),np.sum(indivquantpvalues[i]>0.05),np.sum(indivquantpvalues[i]>0.01),np.sum(indivquantpvalues[i]>0.005),np.sum(indivquantpvalues[i]>0.001),np.mean(indivquantpvalues[i]),np.quantile(indivquantpvalues[i],np.array([0.25,0.5,0.75])))
    for i in range(len(betaquantpvalues)):#####
        print(np.sum(betaquantpvalues[i]>0.1),np.sum(betaquantpvalues[i]>0.05),np.sum(betaquantpvalues[i]>0.01),np.sum(betaquantpvalues[i]>0.005),np.sum(betaquantpvalues[i]>0.001),np.mean(betaquantpvalues[i]),np.quantile(betaquantpvalues[i],np.array([0.25,0.5,0.75])))#####
    print(np.sum(meanpvalues>0.1),np.sum(meanpvalues>0.05),np.sum(meanpvalues>0.01),np.sum(meanpvalues>0.005),np.sum(meanpvalues>0.001),np.mean(meanpvalues),np.quantile(meanpvalues,np.array([0.25,0.5,0.75])))
    print(np.sum(quantpvalues>0.1),np.sum(quantpvalues>0.05),np.sum(quantpvalues>0.01),np.sum(quantpvalues>0.005),np.sum(quantpvalues>0.001),np.mean(quantpvalues),np.quantile(quantpvalues,np.array([0.25,0.5,0.75])))

print(np.sum(indivquantpvalues[0]>0.1)/insts,np.sum(meanpvalues>0.1)/insts,np.sum(quantpvalues>0.1)/insts)
print(np.sum(indivquantpvalues[0]>0.05)/insts,np.sum(meanpvalues>0.05)/insts,np.sum(quantpvalues>0.05)/insts)
print(np.sum(indivquantpvalues[0]>0.01)/insts,np.sum(meanpvalues>0.01)/insts,np.sum(quantpvalues>0.01)/insts)
print(np.sum(indivquantpvalues[0]>0.005)/insts,np.sum(meanpvalues>0.005)/insts,np.sum(quantpvalues>0.005)/insts)
print(np.sum(indivquantpvalues[0]>0.001)/insts,np.sum(meanpvalues>0.001)/insts,np.sum(quantpvalues>0.001)/insts)
print(np.mean(indivquantpvalues[0]>quantpvalues),np.mean(indivquantpvalues[0]<quantpvalues))
print(np.mean(meanpvalues>quantpvalues),np.mean(meanpvalues<quantpvalues))

for i in range(len(indivquantpvalues)):
    print(np.mean(indivquantpvalues[i]))
for i in range(len(betaquantpvalues)):
    print(np.mean(betaquantpvalues[i]))
print(np.mean(meanpvalues),np.mean(quantpvalues))
