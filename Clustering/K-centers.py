def dist(a,b): #ord_norm: none,1,'fro'
    return np.linalg.norm(a - b)

def InitialCenters(X,K):
    
    Y=[] #list for storing centers
    first_c_position= np.random.randint(X.shape[0]) # randomly select a position within obersavation
    Y.append(X[first_c_position]) # add the randomly selected observation to center list
    label = np.zeros(len(X)) # index matrix
    phi = np.zeros(len(X))
    
    """
    search for another (k-1) centers from observation
    """
    for i in range(K-1):
        
        distance=[]
        #phi=0
        
        """
        1. find the nearest center for each observation
        2. calculate d(xi,cj)
        """
        for i in range(len(X)): # for each observation
            min=9999
            for j in range(len(Y)): # compare distance with each center
                d = dist(X[i], Y[j])
                if d < min:
                    min=d
                    #phi[i] = distances # store the distance to the closest center for each observation
                    #index[i] = j
            distance.append(min)  # store the distance to the closest center for each observation (by order)
        
        """
        select the observation with the biggest distance as the next center
        """
        max_dist = max(distance)
        for n in range(len(distance)):
            if distance[n]==max_dist:
                next_c_position = n
        Y.append(X[next_c_position])
        
        distance=[]
    
    """
    Grouping
    """
    for i in range(len(X)): # for each observation
        min_val=9999
        for j in range(0,K): # compare distance with each center
            distances = dist(X[i], Y[j])
            if distances < min_val:
                min_val=distances
                phi[i] = min_val # store the distance to the closest center for each observation
                label[i] = j
    obj=max(phi)
    #max_dist
    
    return Y, label, obj

def Greedy_K_centers (X,K):
    
    iteration = 20
    Y_store=[0]*iteration
    obj_store=[0]*iteration
    label_store=[0]*iteration
    colors = ['r', 'g', 'b','y', 'c', 'm','orange','purple']
    
    for i in range (0,iteration):
        d=999
        Y,label,obj=InitialCenters(X,K)
        
        Y_store[i]=Y
        label_store[i]=label
        obj_store[i]=obj
    
    
    min_d=9999
    index=0
    for j in range(len(obj_store)):
        if obj_store[j]< min_d:
            min_d = obj_store[j]
            index=j
            
    Y_best=np.array(Y_store[j])
    label_best=np.array(label_store[j])
    
    """
    plt.scatter(X[:, 0], X[:, 1], c='b')
    plt.scatter(Y_best[:, 0], Y_best[:, 1], marker='*', s=150, c='k')
    plt.title('Best Initial K-centers (k=%d, iteration :20)'%(len(Y)))
    plt.show()
    """
    
    for z in range (K):
        data=np.array([X[i] for i in range (len(label_best)) if label_best[i]==z])
        plt.scatter(data[:, 0], data[:, 1],s=5, c=colors[z])
        plt.scatter(Y_best[:, 0], Y_best[:, 1], marker='*', s=150, c='k')
        plt.title('Best Initial K-centers (k=%d, iteration :20)'%(len(Y)))
        plt.savefig('K-centers (k=%d, iteration :25)'%(len(Y)))
    plt.show()
    
    
    return Y_best, min_d