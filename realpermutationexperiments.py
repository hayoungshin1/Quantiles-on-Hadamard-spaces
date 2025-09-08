n=2
m=4
origin=torch.zeros(1,3)
origin[0,0]=1
originradial= torch.zeros(m,3)
angles=2*torch.arange(0,m)*math.pi/m
originradial[:,1]=torch.cos(angles)
originradial[:,2]=torch.sin(angles)
betas=torch.tensor([0.8])

data=np.load('moignard_poincare_embedding.npz')
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
medpvalues=np.array([])
quantpvalues=np.array([])
newquantpvalues=np.array([])

distr1=x[y==3]
distr2=x[y==4]

insts=1000
draws=120
tolerance=1e-6
for inst in range(insts):
    np.random.seed(inst)
    x1=distr1[np.random.choice(distr1.shape[0],draws),:]
    x2=distr2[np.random.choice(distr2.shape[0],draws),:]
    x1quantiles=quantile(x1, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
    x2quantiles=quantile(x2, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
    medstat=torch.sum(mag(newlog(x1quantiles,x2quantiles))).item()
    for i in range(len(betas)):
        for j in range(len(originradial)):
            x1quantiles=torch.concat((x1quantiles,quantile(x1, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)),dim=0)
            x2quantiles=torch.concat((x2quantiles,quantile(x2, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)),dim=0)
    quantstat=torch.sum(mag(newlog(x1quantiles,x2quantiles))).item()
    medsum=0
    quantsum=0
    newquantsum=0
    totalx=torch.concat((x1,x2),0)
    #reps=5000
    reps=10000
    for k in range(reps):
        permindices=np.random.choice(totalx.shape[0],x1.shape[0],replace=False)
        permx1=totalx[permindices,:]
        permx2=np.delete(totalx,permindices,axis=0)
        permx1quantiles=quantile(permx1, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
        permx2quantiles=quantile(permx2, 0, origin, torch.unsqueeze(originradial[0,:],0),tol=tolerance)
        permmedstat=torch.sum(mag(newlog(permx1quantiles,permx2quantiles))).item()
        for i in range(len(betas)):
            for j in range(len(originradial)):
                permx1quantiles=torch.concat((permx1quantiles,quantile(permx1, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)),dim=0)
                permx2quantiles=torch.concat((permx2quantiles,quantile(permx2, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0),tol=tolerance)),dim=0)
        permquantstat=torch.sum(mag(newlog(permx1quantiles,permx2quantiles))).item()
        medsum+=(permmedstat>=medstat)
        quantsum+=(permquantstat>=quantstat)
        print(inst,k,medsum,quantsum,newquantsum)
    medpvalues=np.append(medpvalues,medsum/reps)
    quantpvalues=np.append(quantpvalues,quantsum/reps)
    print(np.sum(medpvalues>0.1),np.sum(medpvalues>0.05),np.sum(medpvalues>0.01),np.sum(medpvalues>0.005),np.sum(medpvalues>0.001),np.mean(medpvalues),np.quantile(medpvalues,np.array([0.25,0.5,0.75])))
    print(np.sum(quantpvalues>0.1),np.sum(quantpvalues>0.05),np.sum(quantpvalues>0.01),np.mean(quantpvalues),np.sum(quantpvalues>0.005),np.sum(quantpvalues>0.001),np.quantile(quantpvalues,np.array([0.25,0.5,0.75])))

print(np.sum(medpvalues>0.1)/insts,np.sum(quantpvalues>0.1)/insts)
print(np.sum(medpvalues>0.05)/insts,np.sum(quantpvalues>0.05)/insts)
print(np.sum(medpvalues>0.01)/insts,np.sum(quantpvalues>0.01)/insts)
print(np.sum(medpvalues>0.005)/insts,np.sum(quantpvalues>0.005)/insts)
print(np.sum(medpvalues>0.001)/insts,np.sum(quantpvalues>0.001)/insts)
print(np.mean(medpvalues),np.mean(quantpvalues))
print(np.mean(medpvalues>quantpvalues),np.mean(medpvalues<quantpvalues))
