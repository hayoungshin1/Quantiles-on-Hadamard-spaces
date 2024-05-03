which='real' # simulated or real data

if which=='sim':
    np.random.seed(10)
    x=np.random.normal(0,0.3,(100,2))
    x=torch.Tensor(x)
    x=B2H(x)
elif which=='real':
    data=np.load('olsson_poincare_embedding.npz')
    x=np.concatenate((data['x_train'],data['x_test']),axis=0)
    y=np.concatenate((data['y_train'],data['y_test']),axis=0)
    x=torch.Tensor(x)
    x=B2H(x)
    #x=x[y==3]

m=24

betas=[0.2,0.4,0.6,0.8]
angles=[k*2*np.pi/m for k in range(m)]
xis=[torch.tensor([[np.cos(angles[k]),np.sin(angles[k])]]) for k in range(m)]

quantiles=quantile(x, 0, xis[0])
for i in range(len(betas)):
    for j in range(len(xis)):
        quantiles=torch.concat((quantiles,quantile(x, betas[i], xis[j])),dim=0)
        print(i,j)

quantiles98=torch.empty(0,quantiles.shape[1])
for j in range(len(xis)):
    quantiles98=torch.concat((quantiles98,quantile(x, 0.98, xis[j])),dim=0)
    print(j)

quartiles=torch.empty(0,quantiles.shape[1])
for j in range(len(xis)):
    quartiles=torch.concat((quartiles,quantile(x, 0.5, xis[j])),dim=0)
    print(j)

outliervectors=4*log(torch.unsqueeze(quantiles[0],0),quartiles)
outlierfence=torch.empty(0,quantiles.shape[1])
for j in range(len(xis)):
    outlierfence=torch.concat((outlierfence,exp(torch.unsqueeze(quantiles[0],0),torch.unsqueeze(outliervectors[j],0))),dim=0)

quantiles=H2B(quantiles)
quantiles98=H2B(quantiles98)
quartiles=H2B(quartiles)
outlierfence=H2B(outlierfence)
x=H2B(x)

# draw figure
f = plt.figure(figsize=(7,7))
ax = plt.gca()

if which=='sim':
    plt.scatter(x.cpu().numpy()[:,0], x.cpu().numpy()[:,1], s=30, c='0', marker = '.')
elif which=='real':
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

if which=='sim':
    plt.scatter(x.cpu().numpy()[:,0], x.cpu().numpy()[:,1], s=30, c='0', marker = '.')
elif which=='real':
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
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
plt.axis('equal')
plt.show(block=False)
