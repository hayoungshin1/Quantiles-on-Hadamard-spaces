b=300
m=24
origin=torch.zeros(1,3)
origin[0,0]=1
originradial= torch.zeros(m,3)
angles=2*torch.arange(0,m)*math.pi/m
originradial[:,1]=torch.cos(angles)
originradial[:,2]=torch.sin(angles)

numlevels=4

for moment in ['dispersion', 'skewness', 'kurtosis', 'sasymmetry']:
    for level in range(numlevels):
        np.random.seed(1)
        temp=np.random.normal(0,1,(b,2))/2
        if moment=='dispersion':
            vecs=temp
            vecs[:,1]*=4/(2**level)
        if moment=='skewness':
            #temp2=np.random.exponential(1, size=(b,2))
            vecs=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*temp**2
            vecs[:,1]/=2
        if moment=='kurtosis':
            vecs=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*(temp**3)
            vecs[:,1]/=2
        if moment=='sasymmetry':
            #temp2=np.random.uniform(-3,3,(b,2))
            #temps2[:,1]/=2
            vecs=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*(temp**3)
        vecs-=geometric_median(vecs)
        copy=torch.Tensor(vecs)
        copy=torch.concat((torch.zeros(b,1),copy),dim=1)
        x=exp(origin, copy)
        #median=quantile(x, 0, origin, torch.unsqueeze(originradial[0,:],0))
        #print(mag(log(origin,median)))
        x=H2B(x)
        # draw figure
        f = plt.figure(figsize=(7,7))
        ax = plt.gca()
        plt.scatter(x.cpu().numpy()[:,0], x.cpu().numpy()[:,1], s=30, c='0', marker = '.')
        circle = plt.Circle((0, 0), 1, color='b', fill=False)
        ax.add_patch(circle)
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        plt.axis('equal')
        plt.show(block=False)


dispersion=[]
skewness=[]
kurtosis=[]
supdispersion=[]
avedispersion=[]
supskewness=[]
aveskewness=[]
supkurtosis=[]
avekurtosis=[]
sasymmetry=[]

colors=['tab:orange','tab:green','tab:red','tab:purple']
for moment in ['dispersion', 'skewness', 'kurtosis', 'sasymmetry']:
    for extreme in ['no', 'yes']:
        # draw figure
        f = plt.figure(figsize=(7,7))
        ax = plt.gca()
        plt.scatter(0, 0, s=30, c='tab:blue', marker = '.')
        for level in range(numlevels):
            color=colors[level]
            np.random.seed(1)
            temp=np.random.normal(0,1,(b,2))/2
            if moment=='dispersion':
                vecs=temp
                vecs[:,1]*=4/(2**level)
            if moment=='skewness':
                #temp2=np.random.exponential(1, size=(b,2))
                vecs=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*temp**2
                vecs[:,1]/=2
            if moment=='kurtosis':
                vecs=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*(temp**3)
                vecs[:,1]/=2
            if moment=='sasymmetry':
                #temp2=np.random.uniform(-3,3,(b,2))
                #temps2[:,1]/=2
                vecs=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*(temp**3)
            vecs-=geometric_median(vecs)
            if extreme=='no':
                if moment=='dispersion':
                    betas=torch.tensor([0.5])
                    for k in range(2):
                        dispersion.append(np.quantile(vecs[:,k],0.5+betas.item()/2)-np.quantile(vecs[:,k],0.5-betas.item()/2))
                elif moment=='skewness':
                    betas=torch.tensor([0.5])
                    for k in range(2):
                        skewness.append((np.quantile(vecs[:,k],0.5+betas.item()/2)+np.quantile(vecs[:,k],0.5-betas.item()/2)-2*np.median(vecs[:,k]))/(np.quantile(vecs[:,k],0.5+betas.item()/2)-np.quantile(vecs[:,k],0.5-betas.item()/2)))
                elif moment=='kurtosis':
                    betas=torch.tensor([0.2,0.8])
                    for k in range(2):
                        kurtosis.append((np.quantile(vecs[:,k],0.5+betas[1].item()/2)-np.quantile(vecs[:,k],0.5-betas[1].item()/2))/(np.quantile(vecs[:,k],0.5+betas[0].item()/2)-np.quantile(vecs[:,k],0.5-betas[0].item()/2)))
                elif moment=='sasymmetry':
                    betas=torch.tensor([0.5])
            if extreme=='yes':
                if moment=='dispersion':
                    betas=torch.tensor([0.98])
                    for k in range(2):
                        dispersion.append(np.quantile(vecs[:,k],0.5+betas.item()/2)-np.quantile(vecs[:,k],0.5-betas.item()/2))
                elif moment=='skewness':
                    betas=torch.tensor([0.98])
                    for k in range(2):
                        skewness.append((np.quantile(vecs[:,k],0.5+betas.item()/2)+np.quantile(vecs[:,k],0.5-betas.item()/2)-2*np.median(vecs[:,k]))/(np.quantile(vecs[:,k],0.5+betas.item()/2)-np.quantile(vecs[:,k],0.5-betas.item()/2)))
                elif moment=='kurtosis':
                    betas=torch.tensor([0.2,0.98])
                    for k in range(2):
                        kurtosis.append((np.quantile(vecs[:,k],0.5+0.98/2)-np.quantile(vecs[:,k],0.5-0.98/2))/(np.quantile(vecs[:,k],0.5+0.2/2)-np.quantile(vecs[:,k],0.5-0.2/2)))
                elif moment=='sasymmetry':
                    betas=torch.tensor([0.98])
            print('dispersion:')
            print(dispersion)
            print('skewness:')
            print(skewness)
            print('kurtosis:')
            print(kurtosis)
            copy=torch.Tensor(vecs)
            copy=torch.concat((torch.zeros(b,1),copy),dim=1)
            x=exp(origin, copy)
            #median=quantile(x, 0, origin, torch.unsqueeze(originradial[0,:],0)) # frechet median
            quantiles=torch.empty(0,3)
            #xis=pt(origin,originradial,median) # xi at frechet median
            for i in range(len(betas)):
                for j in range(m):
                    quantiles=torch.concat((quantiles,quantile(x, betas[i].item(), origin, torch.unsqueeze(originradial[j,:],0))),dim=0)
                    print(i,j)
            lift=log(origin,quantiles)
            interranges=torch.zeros(int(m/2))
            for j in range(int(m/2)):
                interranges[j]=mag(torch.unsqueeze(lift[j,:]-lift[j+int(m/2),:],0))
            supinterrange=torch.max(interranges).item()
            aveinterrange=torch.mean(interranges).item()
            opps=torch.zeros(int(m/2),3)
            if moment=='dispersion':
                supdispersion.append(supinterrange)
                avedispersion.append(aveinterrange)
            elif moment=='skewness':
                for j in range(int(m/2)):
                    opps[j,:]=lift[j,:]+lift[j+int(m/2),:]
                supskewness.append(torch.max(mag(opps)).item()/supinterrange)
                aveskewness.append((mag(torch.unsqueeze(torch.mean(opps,0)/2,0))/aveinterrange).item())
            elif moment=='kurtosis':
                for j in range(int(m/2)):
                    opps[j,:]=lift[j,:]+lift[j+int(m/2),:]
                supkurtosis.append(torch.max(mag(opps)).item()/supinterrange)
                avekurtosis.append(torch.mean(mag(opps)).item()/aveinterrange)
            elif moment=='sasymmetry':
                sasymmetry.append(torch.abs(torch.log(torch.max(mag(lift))/torch.min(mag(lift)))).item())
            quantiles=H2B(quantiles)
            for i in range(len(betas)):
                plt.scatter(quantiles[i*m:((i+1)*m)].cpu().numpy()[:,0], quantiles[i*m:((i+1)*m)].cpu().numpy()[:,1], s=30, c=color, marker = '.')
                plt.plot(np.append(quantiles[i*m:((i+1)*m)].cpu().numpy()[:,0],quantiles[i*m:((i+1)*m)].cpu().numpy()[:,0][0])
            , np.append(quantiles[i*m:((i+1)*m)].cpu().numpy()[:,1],quantiles[i*m:((i+1)*m)].cpu().numpy()[:,1][0])
            , c=color)
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
        circle = plt.Circle((0, 0), 1, color='b', fill=False)
        ax.add_patch(circle)
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        plt.axis('equal')
        plt.show(block=False)
