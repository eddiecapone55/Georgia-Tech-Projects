""""""  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Edward Capone  		  	   		 	 	 			  		 			     			  	 
GT User ID: ecapone3  		  	   		 	 	 			  		 			     			  	 
GT ID: 904057332		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import warnings  			 	 			  		 			     			  	 	 	 	 			  		 			     			  	 
import numpy as np

class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None


    def author(self):
        return "ecapone3" 

    def study_group(self):
        return "ecapone3"   

    def add_evidence(self, data_x, data_y):
        # Combine features and target into one data array.
        data = np.hstack((data_x, np.array(data_y).reshape(-1, 1)))
        self.tree = self.build_tree(data)
        
        if self.verbose:
            if self.verbose:
                print(f"Final Decision Tree:\n{self.tree}")

    def build_tree(self, data):
        # Base case 1: If the number of samples is below the leaf size, return a leaf node.
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])
        # Base case 2: If all target values are the same, return a leaf node.
        if np.all(data[:, -1] == data[0, -1]):
            return np.array([[-1, data[0, -1], np.nan, np.nan]])

        # Choose the best feature to split on using the absolute correlation heuristic.
        correlations = []
        for i in range(data.shape[1] - 1):
            if np.std(data[:, i]) == 0:
                correlations.append(0)
            else:
                corr = np.corrcoef(data[:, i], data[:, -1])[0, 1] if np.std(data[:, i]) > 0 else 0
                correlations.append(abs(corr))
        valid_features = [i for i, corr in enumerate(correlations) if corr > 0]
        if not valid_features:
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])
        best_feature = max(valid_features, key=lambda i: correlations[i])

        split_val = np.median(data[:, best_feature])

        # If the split value fails to partition the data, return a leaf.
        if np.all(data[:, best_feature] <= split_val) or np.all(data[:, best_feature] > split_val):
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

        # Partition the data.
        left_mask = data[:, best_feature] <= split_val
        right_mask = data[:, best_feature] > split_val

        left_tree = self.build_tree(data[left_mask])
        right_tree = self.build_tree(data[right_mask])

        # The current root node is stored as: [feature index, split value, left subtree offset, right subtree offset].
        root = np.array([[best_feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        results = np.zeros(points.shape[0])
        for i, point in enumerate(points):
            index = 0
            while self.tree[index, 0] != -1:
                feature = int(self.tree[index, 0])
                index += int(self.tree[index, 2] if point[feature] <= self.tree[index, 1] else self.tree[index, 3])
            results[i] = self.tree[index, 1]
        return results

