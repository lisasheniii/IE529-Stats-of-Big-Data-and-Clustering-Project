def dist(a,b):
    return np.linalg.norm(a-b)

# Gaussian Similarity Function
def Gaussian_Similarity(x1,x2,sigma):
    return math.exp(-(dist(x1,x2))/(2*(sigma**2)))

# Similarity Matrix
def Similarity_Matrix(X,sigma):
    
    N,d = X.shape
    S = np.zeros((N,N))
    
    for i in range (N):
        for j in range (i,N):
            s = Gaussian_Similarity(X[i],X[j],sigma)
            S[i][j]=s
            S[j][i]=s
    
    return S

def Spectral_Clustering(X,K,sigma):
    
    N,d = X.shape
    #S = np.zeros((N,N)) # Gaussian Similarity matrix (euclidean) 
    W = np.zeros((N,N)) # adjacency matrix
    C = np.zeros(N)
    #Y = 
    
    """
    S: Similarity matrix: store the gaussian similarity value between each data point
    """
    S=Similarity_Matrix(X,sigma) # Gaussian Similarity matrix (euclidean)
        
    """
    A: K-nearest neighborhood structure (knn), in order to determine if there is an edge between nodes or not
    """
    # W = kneighbors_graph(X, n_neighbors=5, metric=S).toarray()
    nrst_neigh = NearestNeighbors(n_neighbors = int(N/K), algorithm = 'ball_tree')
    nrst_neigh.fit(X)
    A = nrst_neigh.kneighbors_graph(X).toarray()
    
    
    """
    W: Weighted Adjacency Matrix [N*N] --> combine S & A
    """
    for i in range(N):
        for j in range (i,N):
            if A[i][j]== 1: #there is an edge between node i and j
                W[i][j]= S[i][j]
                W[j][i]= W[i][j]
            else:
                W[i][j]= 0
                W[j][i]= W[i][j]
    
    """
    D matrix [N*N], degreeness
    """
    diag = np.sum(W, axis=1) #sum each value of each row in adjacency matrix
    D = np.diag(diag) 
    
    """
    Liplacian matrix [N*N]
    """
    L = D-W
    
    
    """
    U: first K eigenvectors [N*K dimension]
    """
    eivals, U = np.linalg.eigh(L)
    U=U[:,-K:]
    
    
    input_data = np.array(U)

    
    #Y,C,D = k_means_tol_1(input_data,K,X)
    
    
    #return S,U,Y,C,D
    
    return W,U