b=300
m=24
origin=torch.zeros(1,3)
origin[0,0]=1
originradial= torch.zeros(m,3)
angles=2*torch.arange(0,m)*math.pi/m
originradial[:,1]=torch.cos(angles)
originradial[:,2]=torch.sin(angles)

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

numlevels=4

for moment in ['dispersion', 'skewness', 'kurtosis', 'sasymmetry']:
    for extreme in ['no', 'yes']:
        for level in range(numlevels):
            np.random.seed(1)
            temp=np.random.normal(0,1,(b,2))/2
            if moment=='dispersion':
                vecs=temp
                vecs[:,1]*=4/(2**level)
            if moment=='skewness':
                vecs=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*temp**2
                vecs[:,1]/=2
            if moment=='kurtosis':
                vecs=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*(temp**3)
                vecs[:,1]/=2
            if moment=='sasymmetry':
                vecs=(level/(numlevels-1))*temp+(1-level/(numlevels-1))*(temp**3)
            vecs-=np.mean(vecs,0)
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
            opps=torch.zeros(int(len(xis)/2))
            if moment=='dispersion':
                supdispersion.append(supinterrange)
                avedispersion.append(aveinterrange)
            elif moment=='skewness':
                for j in range(int(len(xis)/2)):
                    opps[j]=mag(torch.unsqueeze(lift[j+1,:]+lift[j+1+int(len(xis)/2),:],0))
                supskewness.append(torch.max(opps).item()/supinterrange)
                aveskewness.append(torch.mean(opps).item()/aveinterrange)
            elif moment=='kurtosis':
                for j in range(int(len(xis)/2)):
                    opps[j]=mag(torch.unsqueeze(lift[j+1+len(xis),:]-lift[j+1+3*int(len(xis)/2),:],0))
                supkurtosis.append(torch.max(opps).item()/supinterrange)
                avekurtosis.append(torch.mean(opps).item()/aveinterrange)
            elif moment=='sasymmetry':
                sasymmetry.append(torch.abs(torch.log(torch.max(mag(lift[1:,:]))/torch.min(mag(lift[1:,:])))).item())
            quantiles=H2B(quantiles)
            x=H2B(x)
            f = plt.figure(figsize=(7,7))
            ax = plt.gca()
            plt.scatter(x.cpu().numpy()[:,0], x.cpu().numpy()[:,1], s=30, c='0', marker = '.')
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


