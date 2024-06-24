import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
        
    if len(data.shape) == 1:
        unique_labels = np.unique(data, return_counts=True)[1]
    else:
        unique_labels = np.unique(data[:,-1], return_counts=True)[1]
    
    for i in (unique_labels):
        gini += (i / len(data))**2

    gini = 1 - gini

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
        
    if len(data.shape) == 1:
        unique_labels = np.unique(data, return_counts=True)[1]
    else:
        unique_labels = np.unique(data[:,-1], return_counts=True)[1]    

    for i in (unique_labels):
        entropy -= (i / len(data)) * np.log2(i / len(data))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:
    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.parent = None
        self.terminal = False # determines if the node is a leaf
        self.chi = chi # holds the chi square p-value (float).
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        y = list(self.data[:,-1])
        pred = max(y, key=y.count)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        self.children.append(node)
        self.children_values.append(val)
        
        node.depth = self.depth + 1 # updating the depth of the child to the depth of the father + 1
        node.parent = self

        if self.terminal == True:
            self.terminal = False # updating the father (current node) to not being a leaf (in case its current status is being a leaf)
        node.terminal = True # updating the child to being a leaf 

        node.max_depth = self.max_depth
        node.chi = self.chi

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.parent is None:
            return None
        else:
            self.feature_importance = (len(self.data) / n_total_sample) * self.parent.goodness_of_split(self.parent.feature)[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        if self.gain_ratio:

            self.impurity_func = calc_entropy

        current_impurity = self.impurity_func(self.data) # calculating the impurity of the original dataset
        feature_impurity = 0 # a variable to calculate the impurity of the dataset after a split

        feature_values = np.unique(self.data[:,feature]) # getting the unique values of the feature
        
        for val in feature_values:
            
            data_subset = self.data[self.data[:,feature] == val] # creating a subset according to one of the values of the feature
            subset_weight = len(data_subset) / len(self.data)  
            feature_impurity += subset_weight * self.impurity_func(data_subset) # adding the impurity of the subset to goodness
            
            groups[val] = data_subset # adding the subset

        if self.gain_ratio:
            if (self.__all_equal(self.data[:,feature])):
                goodness=0
            else:
                goodness = (current_impurity - feature_impurity) / self.impurity_func(self.data[:,feature])

        else:
        
            goodness = current_impurity - feature_impurity

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups
    
    def __all_equal(self, arr):
        return all(x == arr[0] for x in arr)
    
    def chi_pruning_condition(self, feature_values):
        try:
            if self.chi == 1:
                return True
            labels = np.unique(self.data[:,-1])
            chi_statstic = 0
            for label in labels:
                for feature_value in feature_values:
                    feature_mask = self.data[:,self.feature] == feature_value
                    label_mask=self.data[:,-1] == label
                    observerd=len(self.data[feature_mask & label_mask])
                    label_count=len(self.data[label_mask])
                    expected=len(self.data[feature_mask]) / len(self.data) * label_count
                    chi_statstic+=((observerd - expected)**2)/expected
            
            # Retrieve chi from chi table
            degrees_of_freedom = len(feature_values) - 1 
            
            return chi_table[degrees_of_freedom][self.chi] < chi_statstic
        except Exception as exeption:
            pass
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        if self.depth == self.max_depth:
            self.terminal = True
            return
        

        goodness = [] # holds the goodness of split for each feature
        
        for i in range(len(self.data[0,:-1])):
        
            goodness.append(self.goodness_of_split(i)[0])

        self.feature = np.argmax(goodness) # takes the feature with the best goodness of split
        
        feature_values = np.unique(self.data[:,self.feature]) # gets the unique values of the feature
        if len(feature_values) <= 1:
            self.terminal = True
            return
        if not self.chi_pruning_condition(feature_values):
            self.terminal = True
            return

        for feature_value in feature_values: 
            self.add_child(DecisionNode(self.data[self.data[:,self.feature] == feature_value], self.impurity_func), feature_value)
    
            
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################        


                    
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Create a decesision node for the root
        self.root = DecisionNode(self.data, self.impurity_func, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)

        # Recursively split the children of the root
        self.split_children(self.root)


        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    def split_children(self, node):
        """
        Recursively splits the children of a node if the node is not terminal.
        This function has no return value
        """
        ###########################################################################
        node.split()
        if node.goodness_of_split(node.feature)[0] == 0:
            return
        for child in node.children:
            self.split_children(child)

        ###########################################################################
    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        
        node = self.root
        try:
            while node.terminal == False:
                node = node.children[node.children_values.index(instance[node.feature])]
        
        except Exception as e:
            pass
            

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # Iterate over the dataset and predict each instance

        correct = 0
        for i in range(len(dataset)):
            if self.predict(dataset[i]) == dataset[i][-1]:
                correct += 1
        
        accuracy = correct / len(dataset) * 100
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        
    def depth(self):
        return self.root.depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        tree = DecisionTree(X_train, calc_gini, max_depth=max_depth, gain_ratio=True)
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for chi in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        tree = DecisionTree(X_train, calc_gini, chi=chi, gain_ratio=True)
        tree.build_tree()
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_test))
        depth.append(tree.root.depth)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    
    n_nodes = 1
    for child in node.children:
        n_nodes += count_nodes(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






