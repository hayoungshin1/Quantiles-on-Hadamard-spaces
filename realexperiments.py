n=2
m=24
origin=torch.zeros(1,3)
origin[0,0]=1
originradial= torch.zeros(m,3)
angles=2*torch.arange(0,m)*math.pi/m
originradial[:,1]=torch.cos(angles)
originradial[:,2]=torch.sin(angles)

which='real'
if which=='sim':
    np.random.seed(10)
    x=np.random.normal(0,0.3,(100,2))
    x=torch.Tensor(x)
    #x[:,1]/=4
    x=B2H(x)
elif which=='real':
    data=np.load('olsson_poincare_embedding.npz')
    x=np.concatenate((data['x_train'],data['x_test']),axis=0)
    y=np.concatenate((data['y_train'],data['y_test']),axis=0)
    x=torch.Tensor(x)
    x=B2H(x)
    #x=x[y==3]

supdispersion=[]
avedispersion=[]
supskewness=[]
aveskewness=[]
supkurtosis=[]
avekurtosis=[]
sasymmetry=[]

numlevels=4

for moment in ['other', 'kurtosis']:
    for extreme in ['no', 'yes']:
        if extreme=='no':
            if moment=='other':
                betas=torch.tensor([0.5])
            elif moment=='kurtosis':
                betas=torch.tensor([0.2,0.8])
        if extreme=='yes':
            if moment=='other':
                betas=torch.tensor([0.98])
                for k in range(2):
            elif moment=='kurtosis':
                betas=torch.tensor([0.2,0.98])
        quantiles=quantile(x, 0, origin, torch.unsqueeze(originradial[0,:],0)) # frechet median
        median=quantiles.detach().clone()
        xis=pt(origin,originradial,median) # xi at frechet median
        for i in range(len(betas)):
            for j in range(len(xis)):
                quantiles=torch.concat((quantiles,quantile(x, betas[i].item(), median, torch.unsqueeze(xis[j,:],0))),dim=0)
                print(i,j)
        lift=log(median,quantiles)
        interranges=torch.zeros(int(len(xis)/2))
        for j in range(int(len(xis)/2)):
            interranges[j]=mag(torch.unsqueeze(lift[j+1,:]-lift[j+1+int(len(xis)/2),:],0))
        supinterrange=torch.max(interranges).item()
        aveinterrange=torch.mean(interranges).item()
        opps=torch.zeros(int(len(xis)/2),3)
        if moment=='other':
            supdispersion.append(supinterrange)
            avedispersion.append(aveinterrange)
            for j in range(int(len(xis)/2)):
                opps[j,:]=lift[j,:]+lift[j+int(len(xis)/2),:]
            supskewness.append(torch.max(mag(opps)).item()/supinterrange)
            aveskewness.append((mag(torch.unsqueeze(torch.mean(opps,0)/2,0))/aveinterrange).item())
            sasymmetry.append(torch.abs(torch.log(torch.max(mag(lift[1:,:]))/torch.min(mag(lift[1:,:])))).item())
        elif moment=='kurtosis':
            for j in range(int(len(xis)/2)):
                opps[j,:]=lift[j,:]+lift[j+int(len(xis)/2),:]
            supkurtosis.append(torch.max(mag(opps)).item()/supinterrange)
            avekurtosis.append(torch.mean(mag(opps)).item()/aveinterrange)
        print('supdispersion:')
        print(supdispersion)
        print('avedispersion:')
        print(avedispersion)
        print('supskewness:')
        print(supskewness)
        print('aveskewness:')
        print(aveskewness)
        print('supkurtosis:')
        print(supkurtosis)
        print('avekurtosis:')
        print(avekurtosis)
        print('sasymmetry:')
        print(sasymmetry)

x=x/torch.unsqueeze(torch.sqrt(-ip(x,x)),1)
frechet=frechetmean(x)
lift=log(frechet,x)
originbasis=torch.concat((torch.zeros(n,1),torch.eye(n)),dim=1) # orthonormal basis at (1,0,...,0)
frechetbasis=pt(origin,originbasis,frechet) # orthonormal basis at frechet mean
newlift=vorth(lift, frechetbasis)
b=x.shape[0]
prec=torch.linalg.inv(torch.matmul(torch.transpose(newlift,0,1),newlift)/b)
ratios=[]
index_list=list(combinations(range(b),2))
for l in range(len(index_list)):
    indices=index_list[l]
    target=torch.matmul(torch.matmul(newlift[indices,:],prec),torch.transpose(newlift[indices,:],0,1))
    ratios.append((torch.trace(target)/n)/(torch.linalg.det(target)**(1/n)))
index, element = min(enumerate(ratios), key=itemgetter(1))
best=index_list[index]
best=newlift[index_list[index],:]
points=exp(frechet,transform(frechet,frechetbasis,newlift,torch.linalg.inv(best))) # transformed data

betas=torch.tensor([0.2,0.4,0.6,0.8])

quantiles=quantile(points, 0, origin, torch.unsqueeze(originradial[0,:],0))
for i in range(len(betas)):
    for j in range(len(originradial)):
        quantiles=torch.concat((quantiles,quantile(points, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0))),dim=0)
        print(i,j)

quantiles98=torch.empty(0,n+1)
for j in range(len(originradial)):
    quantiles98=torch.concat((quantiles98,quantile(points, 0.98, origin, torch.unsqueeze(originradial[j,:],0))),dim=0)
    print(j)

quartiles=torch.empty(0,n+1)
for j in range(len(originradial)):
    quartiles=torch.concat((quartiles,quantile(points, 0.5, origin, torch.unsqueeze(originradial[j,:],0))),dim=0)
    print(j)

outlierfence=exp(torch.unsqueeze(quantiles[0],0),4*log(torch.unsqueeze(quantiles[0],0),quartiles))


qlift=log(frechet,quantiles)
newqlift=vorth(qlift,frechetbasis)
quantiles=exp(frechet,transform(frechet,frechetbasis,newqlift,best)) # retransformed quantiles
qlift=log(frechet,quantiles98)
newqlift=vorth(qlift,frechetbasis)
quantiles98=exp(frechet,transform(frechet,frechetbasis,newqlift,best)) # retransformed quantiles
qlift=log(frechet,quartiles)
newqlift=vorth(qlift,frechetbasis)
quartiles=exp(frechet,transform(frechet,frechetbasis,newqlift,best)) # retransformed quantiles
qlift=log(frechet,outlierfence)
newqlift=vorth(qlift,frechetbasis)
outlierfence=exp(frechet,transform(frechet,frechetbasis,newqlift,best)) # retransformed quantiles

quantiles=H2B(quantiles)
quantiles98=H2B(quantiles98)
quartiles=H2B(quartiles)
outlierfence=H2B(outlierfence)
x=H2B(x)

# draw figure
f = plt.figure(figsize=(7,7))
ax = plt.gca()

colors=['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7']
for j in range(8):
    plt.scatter(x[y==j].cpu().numpy()[:,0], x[y==j].cpu().numpy()[:,1], s=10, c=colors[j], marker = '.')


plt.scatter(quantiles[0].cpu().numpy()[0], quantiles[0].cpu().numpy()[1], s=30, c='tab:blue', marker = '.')
for i in range(len(betas)):
    plt.scatter(quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,0], quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,1], s=30, c='tab:blue', marker = '.')
    plt.plot(np.append(quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,0],quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,0][0])
, np.append(quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,1],quantiles[(i*m+1):((i+1)*m+1)].cpu().numpy()[:,1][0])
, c='tab:blue')

circle = plt.Circle((0, 0), 1, color='b', fill=False)
ax.add_patch(circle)
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
plt.axis('equal')
plt.show(block=False)

# draw outlier figure
f = plt.figure(figsize=(7,7))
ax = plt.gca()

colors=['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7']
for j in range(8):
    plt.scatter(x[y==j].cpu().numpy()[:,0], x[y==j].cpu().numpy()[:,1], s=10, c=colors[j], marker = '.')


plt.scatter(quantiles98.cpu().numpy()[:,0], quantiles98.cpu().numpy()[:,1], s=30, c='tab:green', marker = '.')
plt.plot(np.append(quantiles98.cpu().numpy()[:,0],quantiles98.cpu().numpy()[:,0][0])
, np.append(quantiles98.cpu().numpy()[:,1],quantiles98.cpu().numpy()[:,1][0])
, c='tab:green')

plt.scatter(quartiles.cpu().numpy()[:,0], quartiles.cpu().numpy()[:,1], s=30, c='tab:orange', marker = '.')
plt.plot(np.append(quartiles.cpu().numpy()[:,0],quartiles.cpu().numpy()[:,0][0])
, np.append(quartiles.cpu().numpy()[:,1],quartiles.cpu().numpy()[:,1][0])
, c='tab:orange')

plt.scatter(outlierfence.cpu().numpy()[:,0], outlierfence.cpu().numpy()[:,1], s=30, c='tab:red', marker = '.')
plt.plot(np.append(outlierfence.cpu().numpy()[:,0],outlierfence.cpu().numpy()[:,0][0])
, np.append(outlierfence.cpu().numpy()[:,1],outlierfence.cpu().numpy()[:,1][0])
, c='tab:red')

circle = plt.Circle((0, 0), 1, color='b', fill=False)
ax.add_patch(circle)
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
plt.axis('equal')
plt.show(block=False)
