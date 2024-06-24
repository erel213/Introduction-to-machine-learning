import numpy as np
import pandas as pd

np.set_printoptions(precision=30, suppress=False)

def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    r = np.corrcoef(x,y)[0,1]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    features_dict = {}
    
    for col in X.select_dtypes(include=['number']).columns: # selecting only columns with numeric data types
      
      features_dict[col] = pearson_correlation(X[col],y)

    n_features_sorted_dict = dict(sorted(features_dict.items(), key=lambda item: item[1], reverse=True)[:n_features])
    best_features = list(n_features_sorted_dict.keys())
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def apply_bias_trick(self, X):

        if X.ndim == 1:
            
            X = X.reshape(X.shape[0], 1)

        X_columns = X.shape[1]
        ones_column = np.ones((X.shape[0], 1))
        
        X = np.hstack((ones_column, X))

        return X

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Initalize random theta vector (weights)
        X = self.apply_bias_trick(X)
        
        self.theta = np.random.rand(X.shape[1]) # initialize theta
        m = X.shape[0] # number of instances

        for i in range(self.n_iter):
          
          # Calculate the probability using the sigmoid function
          p = 1/(1 + np.exp(-X @ self.theta))

          # Calculate the gradient
          gradient = X.T @ (p-y)

          # Update theta
          self.theta = self.theta - self.eta * gradient

          cost = -1/m * np.sum(y@np.log(p) + (1-y)@np.log(1-p))
          # Add history calculation for cost function and thetas
          self.Js.append(cost)
          self.thetas.append(self.theta)

          # Check for convergence
          if i > 0 and abs(self.Js[i-1] - self.Js[i]) < self.eps:
            break

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = self.apply_bias_trick(X)

        p = 1 / (1 + np.exp(-X @ self.theta))

        preds = (p >= 0.5).astype(int)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    # Split the indices to folds
    fold_sizes = np.full(folds, X.shape[0] // folds, dtype=int)
    fold_sizes[:X.shape[0] % folds] += 1
    current = 0
    
    folds_indices = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds_indices.append(indices[start:stop])
        current = stop

    fold_accuracy = []

    # Create the training and validation sets for each fold
    for i in range(folds):
        validation_indices = folds_indices[i]
        training_indices = np.hstack([folds_indices[j] for j in range(folds) if j != i])
        
        X_train, X_val = X[training_indices], X[validation_indices]
        y_train, y_val = y[training_indices], y[validation_indices]

        algo.fit(X_train, y_train)
        y_pred = algo.predict(X_val)
        fold_accuracy.append((np.sum(y_pred == y_val))/len(y_val))

    cv_accuracy = np.mean(fold_accuracy)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    epsilon = 1 * 10 ** -5
    numerator = epsilon if np.any(np.exp(-((data - mu) ** 2) / (2 * sigma ** 2)) == 0) else np.exp(-((data - mu) ** 2) / (2 * sigma ** 2))
    denominator = np.sqrt(2 * np.pi * sigma ** 2)
    p = numerator / denominator
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    def generate_random_numbers_sum_to_one(self, k):
        # Generate k random numbers
        random_numbers = np.random.rand(k)
        
        # Normalize the numbers so that they sum to 1
        random_numbers /= random_numbers.sum()
        
        return random_numbers
        
        # initial guesses for parameters
    
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################        
        self.weights = self.generate_random_numbers_sum_to_one(self.k)
        self.mus = np.random.rand(self.k)
        self.sigmas = np.random.rand(self.k)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.responsibilities = np.zeros((data.shape[0], self.k))

        for i in range(data.shape[0]):
          for j in range(self.k):
            self.responsibilities[i,j] = self.weights[j] * norm_pdf(data[i], self.mus[j], self.sigmas[j])

        # Normalize the responsibilities
        self.responsibilities = self.responsibilities / np.sum(self.responsibilities, axis=1)[:, None]
                
      
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.weights = self.responsibilities.sum(axis = 0) / len(data)
        self.mus = np.sum(self.responsibilities * data.reshape(-1,1), axis = 0) / (self.weights * len(data))
        self.sigmas = np.sqrt(np.sum(self.responsibilities * np.square(data.reshape(-1,1) - self.mus), axis = 0) / (self.weights * len(data)))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        self.costs = []

        for iter in range(self.n_iter):
            
            self.expectation(data)
            self.maximization(data)
            
            likelihood = []
            
            for x in data:
                for j in range(len(self.weights)):    
                  likelihood.append(-np.log(self.weights[j]) - np.log(norm_pdf(x, self.mus[j], self.sigmas[j])))
            
            self.costs.append(np.sum(likelihood))
                        
            # Check for convergence
            if iter > 0 and abs(self.costs[iter-1] - self.costs[iter]) < self.eps:
              break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################   
    k = len(weights)

    pdf = np.zeros_like(data)
    for i in range(k):
        pdf += weights[i] * norm_pdf(data, mus[i], sigmas[i])
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        
        self.class_datasets = []
        self.prior = None
        self.weights = []
        self.mus = []
        self.sigmas = []
        self.class_values = None
    
    def create_class_datasets(self, X, y):
        
        # create a unified dataset
        dataset = np.hstack([X, y.reshape([-1,1])]) 
        
        class_values = np.unique(y) 
        
        # seperate to different datasets by class value

        for i, class_value in enumerate(class_values):
            self.class_datasets.append(dataset[dataset[:,-1] == class_value])

    def create_dist_params(self, X):
        
        for i in range(X.shape[1]):
            
            model = EM(k = self.k)
            model.fit(X[:,i])
            self.weights.append(model.get_dist_params()[0])
            self.mus.append(model.get_dist_params()[1])
            self.sigmas.append(model.get_dist_params()[2])

    def get_instance_likelihood(self, x, weights, mus, sigmas):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
       
        feature_likelihoods = gmm_pdf(x, weights, mus, sigmas)
        
        likelihood = np.prod(feature_likelihoods)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.class_values = np.unique(y)

        # seperate data by classes
        self.create_class_datasets(X, y)

        # calculate prior for each class
        self.prior = [len(self.class_datasets[i])/len(y) for i in range(len(self.class_datasets))]

        for i in self.class_datasets:
            
          self.create_dist_params(i[:,:-1])

        # calculate likelihood for each class


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        preds = []
        for x in X:
            posteriors = []
            for i in range(len(self.class_values)):
                prior = self.prior[i]
                likelihood = self.get_instance_likelihood(x, weights= self.weights[i], mus=self.mus[i], sigmas=self.sigmas[i])
                posteriors.append(prior * likelihood)
            preds.append(self.class_values[np.argmax(posteriors)])
        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Logistic Regression
    lr = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lr.fit(x_train, y_train)
    lor_train_acc = np.mean(lr.predict(x_train) == y_train)
    lor_test_acc = np.mean(lr.predict(x_test) == y_test)

    # Naive Bayes
    nb = NaiveBayesGaussian(k=k)
    nb.fit(x_train, y_train)
    bayes_train_acc = np.mean(nb.predict(x_train) == y_train)
    bayes_test_acc = np.mean(nb.predict(x_test) == y_test)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Generate dataset for Naive Bayes
    mean1_a = [1, 2, 3]
    cov1_a = [[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]]
    mean2_a = [7, 8, 9]
    cov2_a = [[1, -0.1, -0.1], [-0.1, 1, -0.1], [-0.1, -0.1, 1]]

    class1_a = multivariate_normal.rvs(mean=mean1_a, cov=cov1_a, size=100)
    class2_a = multivariate_normal.rvs(mean=mean2_a, cov=cov2_a, size=100)

    labels1_a = np.zeros((100, 1))
    labels2_a = np.ones((100, 1))

    dataset_a_features = np.vstack((class1_a, class2_a))
    dataset_a_labels = np.vstack((labels1_a, labels2_a))

    # Generate dataset for Logistic Regression
    mean1_b = [1, 1, 1]
    cov1_b = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mean2_b = [5, 5, 5]
    cov2_b = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    class1_b = multivariate_normal.rvs(mean=mean1_b, cov=cov1_b, size=100)
    class2_b = multivariate_normal.rvs(mean=mean2_b, cov=cov2_b, size=100)

    labels1_b = np.zeros((100, 1))
    labels2_b = np.ones((100, 1))

    dataset_b_features = np.vstack((class1_b, class2_b))
    dataset_b_labels = np.vstack((labels1_b, labels2_b))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }