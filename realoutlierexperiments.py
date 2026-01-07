n=2
m=120
origin=torch.zeros(1,3)
origin[0,0]=1
originradial= torch.zeros(m,3)
angles=2*torch.arange(0,m)*math.pi/m
originradial[:,1]=torch.cos(angles)
originradial[:,2]=torch.sin(angles)

robust='No' # robust='Yes'
outliers='No' # outliers='Yes'

data=np.load('Downloads/olsson_poincare_embedding.npz')
xdata=np.concatenate((data['x_train'],data['x_test']),axis=0)
ydata=np.concatenate((data['y_train'],data['y_test']),axis=0)
labels=[3]
x=xdata[ydata==[labels[0]]]
y=ydata[ydata==[labels[0]]]
if outliers=='Yes':
    labels.append(6)
    x=np.concatenate((x,xdata[ydata==labels[1]]),0)
    y=np.concatenate((y,ydata[ydata==labels[1]]),0)
x=torch.Tensor(x)
x=B2H(x)

x=x/torch.unsqueeze(torch.sqrt(-ip(x,x)),1)
b=x.shape[0]
originbasis=torch.concat((torch.zeros(n,1),torch.eye(n)),dim=1) # orthonormal basis at (1,0,...,0)

frechet=quantile(x, 0, origin, torch.unsqueeze(originradial[0,:],0))
lift=log(frechet,x)
frechetbasis=pt(origin,originbasis,frechet) # orthonormal basis at frechet median
newlift=vorth(lift, frechetbasis)
if robust=='No':
    prec=torch.linalg.inv(torch.Tensor(EmpiricalCovariance().fit(newlift.numpy()).covariance_))
if robust=='Yes':
    prec=torch.linalg.inv(torch.Tensor(MinCovDet().fit(newlift.numpy()).covariance_))
ratios=[]
index_list=list(combinations(range(b),n))
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

quartiles=torch.empty(0,n+1)
for j in range(len(originradial)):
    quartiles=torch.concat((quartiles,quantile(points, 0.5, origin, torch.unsqueeze(originradial[j,:],0))),dim=0)
    print(j)

outlierfence=exp(torch.unsqueeze(quantiles[0],0),3*log(torch.unsqueeze(quantiles[0],0),quartiles))


qlift=log(frechet,quantiles)
newqlift=vorth(qlift,frechetbasis)
quantiles=exp(frechet,transform(frechet,frechetbasis,newqlift,best)) # retransformed quantiles
qlift=log(frechet,quartiles)
newqlift=vorth(qlift,frechetbasis)
quartiles=exp(frechet,transform(frechet,frechetbasis,newqlift,best)) # retransformed quantiles
qlift=log(frechet,outlierfence)
newqlift=vorth(qlift,frechetbasis)
outlierfence=exp(frechet,transform(frechet,frechetbasis,newqlift,best)) # retransformed quantiles

quantiles=H2B(quantiles)
quartiles=H2B(quartiles)
outlierfence=H2B(outlierfence)
x=H2B(x)

# draw figure
f = plt.figure(figsize=(7,7))
ax = plt.gca()

colors=['0','m']
for j in range(len(labels)):
    plt.scatter(x[y==labels[j]].cpu().numpy()[:,0], x[y==labels[j]].cpu().numpy()[:,1], s=10, c=colors[j], marker = '.')


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

colors=['0','m']
for j in range(len(labels)):
    plt.scatter(x[y==labels[j]].cpu().numpy()[:,0], x[y==labels[j]].cpu().numpy()[:,1], s=10, c=colors[j], marker = '.')

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
