def k_means_tol_1(X,K): # X: d-dimensional obersvation, K: number of clusters
    index = np.zeros(len(X)) # index matrix
    phi = np.zeros(len(X))
    
    tol= 10**(-5) #convergence threshold
    Y = np.array(random.sample(list(X),K)) # randomly select K oberservation from dataset to set as our initial centers
    Y_old = np.zeros(Y.shape) # new centers awaiting assignment
    con = np.linalg.norm(Y-Y_old,ord=1) # 1-norm btw Y and Y_new
    avg_D=0 # Obj value
    colors = ['r', 'g', 'b','y', 'c', 'm','orange','purple'] # color pallete for plotting cluster
    
    while con > tol:
        
        """
        Assign data to the closest center (with label)
        """
        for i in range(len(X)): # for each observation
            min=9999
            for j in range(0,K): # compare distance with each center
                distances = dist(X[i], Y[j])
                if distances < min:
                    min=distances
                    phi[i] = distances # store the distance to the closest center for each observation
                    index[i] = j # store the index (label) for each observation to locate its closest center
        
        Y_old = deepcopy(Y)
        
        """
        Compute cluster mean and assign it as the new center
        """
        for j in range (K):
            group_data_point=[X[i] for i in range (len(index)) if index[i]==j]
            Y[j] = sum(group_data_point)/len(group_data_point)
            con = np.linalg.norm(Y-Y_old,ord=1)
            #print(con)
            
        """
        D (objective value)
        """
        avg_D = sum(phi)/len(phi)
    
    """
    Plot with color by clusters
    """
    for z in range (K):
        data=np.array([X[i] for i in range (len(index)) if index[i]==z])
        plt.scatter(data[:, 0], data[:, 1],s=10, c=colors[z])
        plt.scatter(Y[:, 0], Y[:, 1], marker='*', s=150, c='k')
        plt.title('K-means (tol= 1e-5, D=%f)'%(avg_D))
        plt.savefig('K-means (tol= 1e-5, K=%d, D=%f).png'%(K,avg_D))
        
    plt.show()
        
    return Y,index,avg_D