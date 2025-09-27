import numpy as np

class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return "ecapone3"  

    def study_group(self):
        return "ecapone3"   
    
    def add_evidence(self, data_x, data_y):
        if data_x.shape[0] == 0 or data_y.shape[0] == 0:
            raise ValueError("Input data cannot be empty")
        data = np.hstack((data_x, np.array(data_y).reshape(-1, 1)))
        self.tree = self.build_tree(data)
        
        if self.verbose:
            print("Final Decision Tree:\n", self.tree)

    def build_tree(self, data):
        # Base case 1.
        if data.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])
        # Base case 2.
        if np.all(data[:, -1] == data[0, -1]):
            return np.array([[-1, data[0, -1], np.nan, np.nan]])

        num_features = data.shape[1] - 1
        # Randomly select a feature.
        feature = np.random.randint(0, num_features)
        feature_values = data[:, feature]
        min_val = np.min(feature_values)
        max_val = np.max(feature_values)

        # If the chosen feature has identical values, return a leaf.
        if min_val == max_val:
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

        # Choose a random split value between min and max.
        split_val = np.random.uniform(min_val, max_val)

        left_mask = data[:, feature] <= split_val
        right_mask = data[:, feature] > split_val

        # If the split fails, return a leaf.
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return np.array([[-1, np.mean(data[:, -1]), np.nan, np.nan]])

        left_tree = self.build_tree(data[left_mask])
        right_tree = self.build_tree(data[right_mask])

        root = np.array([[feature, split_val, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        results = np.zeros(points.shape[0])
        for i in range(points.shape[0]):
            index = 0
            while self.tree[index, 0] != -1:
                feature = int(self.tree[index, 0])
                index += int(self.tree[index, 2] if points[i, feature] <= self.tree[index, 1] else self.tree[index, 3])
            results[i] = self.tree[index, 1]
        return results
