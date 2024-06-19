import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    row_idx = np.random.choice(a = X.shape[0], size = k, replace = False)
    centroids = X[row_idx]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float64) 

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for centroid in centroids:

        distances.append((np.abs(X - centroid) ** p).sum(axis = 1))

    distances = np.asarray(distances)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    previous_centroids = centroids
    for i in range(max_iter):

        distances = lp_distance(X, centroids, p)
        classes = distances.argmin(axis = 0)

        for j, centroid in enumerate(centroids):

            centroid[:] = X[classes == j].mean(axis = 0)

        if np.all(centroids == previous_centroids):
            break

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    centroids = []
    centroids.append(get_random_centroids(X, 1))  # choosing a uniformly random centroid
        
    for i in range(k-1):
        
        raw_distances = lp_distance(X, centroids, p) # compute the distances of each point to all centroids
        distances = raw_distances.min(axis = 0) # compute the distance of each point to its nearest centroid
        squared_distances = distances ** 2 
        total_squared_distances = squared_distances.sum()
        weights = squared_distances / total_squared_distances
        row_idx = np.random.choice(a = X.shape[0], size = 1, p = weights)
        centroids.append(X[row_idx])
    
    centroids = np.asarray(centroids).astype(np.float64).reshape(k,3)
    centroids, classes = kmeans(X, k, p, max_iter)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes
