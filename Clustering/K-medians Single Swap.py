def dist(a,b): #ord_norm: none,1,'fro'
    return np.linalg.norm(a - b)

def InitialKCenters(X,K):
    
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
    
    #return Y, label, obj
    
    return Y, label

def all_cost(X,Q): # X:data, Q:medians
    
    N,d = X.shape
    n_k = np.array(Q).shape[0]
    
    D = np.zeros((N,n_k))
    
    for i in range (X.shape[0]):
        for j in range (len(Q)):
            cost= dist(X[i],Q[j])
            D[i][j]=cost
        
    return D

def Label(D): # input: the output of cost function
    
    N,kk = D.shape
    labels=np.zeros(N)
    obj_cost=[] #
    
    for i in range (N):
        min=999
        for j in range (kk):
            c= D[i][j]
            if c < min:
                min = c
            #labels[i] = j
                labels[i] = j
        obj_cost.append(min) #
        
    sum_cost= sum(obj_cost) #
        
    return labels, sum_cost

def single_swap_debug(X,Q,tau): 
    
    """
    Assign each data point to the new cluster 
    - calculating the cost(distance) for <all observations> 
    - reassign labels
    """
    D = all_cost(X,Q)
    label,sum_cost = Label(D) #
    
    new_Q = Q
    
    for i in range(len(Q)):
        # collect data points in cluster(i) as a new list
        cluster_data_0 = np.array([X[j] for j in range (len(label)) if label[j]==i])
        
        # calculate the cost matrix for each cluster (median)
        cluster_cost = all_cost(cluster_data_0,Q[i])
        cost_Qi = np.sum(cluster_cost) 
            
    
        """
        Calculate the new cost:
        by assigning each data points in cluster (i) other than median (i) as the new median
        """
        for a in cluster_data_0: # grab each data points in cluster (i) by iteration
            new_Qi = a
            new_cluster_cost = all_cost(cluster_data_0,a) ## ??? how to exclude [a] in [cluster_data]
            new_cost_Qi = np.sum(new_cluster_cost)
            
            """
            Compare new cost with the old one:
            if it reduce by (1-tau), update median(i)
            """
            if new_cost_Qi <= (1-tau)*cost_Qi :
                cost_Qi =  new_cost_Qi
                
                new_Q[i] = a # update median(i)
            
            #else:
            #    new_Q[i] = Q[i]
    
    new_D = all_cost(X,new_Q)
    new_label, new_sum_cost = Label(new_D) #
    
    return new_Q, new_label, new_sum_cost

def convg_optimal(Q,new_Q):
    
    #use set() to hash the tuple and sort it
    Q_val = set([tuple(c) for c in Q]) 
    new_Q_val = set([tuple(c) for c in new_Q])
    
    """
    True: medians does not move after swaps
    False: medians moved
    """
    return Q_val == new_Q_val

def Kcenter_improve_by_Kmedians_Single_Swap(X,K,tau,max_same): #debug (test which swap is correct)
    
    colors = ['r', 'g', 'b','y', 'c', 'm','orange','purple']
    
    """
    Generate the initial k-centers
    """
    first_Q,first_label = InitialKCenters(X,K)
    
    
    for f in range (K):
        data=np.array([X[i] for i in range (len(first_label)) if first_label[i]==f])
        plt.scatter(data[:, 0], data[:, 1],s=5, c=colors[f])
        plt.scatter(np.array(first_Q)[:, 0], np.array(first_Q)[:, 1], marker='*', s=150, c='k')
        plt.title('Kcenter (k=%d)'%(len(first_Q)))
        #plt.savefig('K-centers (k=%d, iteration :25)'%(len(Y)))
    plt.show()
    
    new_Q = first_Q.copy()
    
    Stop = False
    i=0
    
    #colors = ['r', 'g', 'b','y', 'c', 'm','orange','purple']
    
    # Stop swapping if the medians did not move for [max_same] iterations
    while (not Stop) and (i <= max_same):
        
        Q = new_Q.copy()
        
        """
        Perform single swap heuristic
        """
        new_Q, new_label, new_sum_cost = single_swap_debug(X,new_Q,tau)
        

        """
        True: medians does not move after swaps
        False: medians moved
        """
        Stop = convg_optimal(Q,new_Q)
        
        i+=1 # count how many times does medians remain the same
        
    
    
    
    for z in range (K):
        data=np.array([X[i] for i in range (len(new_label)) if new_label[i]==z])
        plt.scatter(data[:, 0], data[:, 1],s=5, c=colors[z])
        plt.scatter(np.array(new_Q)[:, 0], np.array(new_Q)[:, 1], marker='*', s=150, c='k')
        plt.title('Kcenter_improve_by_Kmedians_Single_Swap (k=%d)'%(len(new_Q)))
        #plt.savefig('K-centers (k=%d, iteration :25)'%(len(Y)))
    plt.show()
    
    
    
    
    return new_Q, new_label, new_sum_cost, first_Q, first_label