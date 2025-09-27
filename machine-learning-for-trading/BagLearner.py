# BagLearner.py
import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost  # Boosting is optional and not implemented in this example.
        self.verbose = verbose
        self.learners = []

    def author(self):
        return "ecapone3" 

    def study_group(self):
        return "ecapone3"   

    def add_evidence(self, data_x, data_y):
        if data_x.shape[0] == 0 or data_y.shape[0] == 0:
            raise ValueError("Input data cannot be empty.")
        n = data_x.shape[0]
        self.learners = []  # Reset learners on new evidence.
        for i in range(self.bags):
            indices = np.random.choice(n, n, replace=True)
            learner_instance = self.learner(**{**self.kwargs, "verbose": self.verbose})
            learner_instance.add_evidence(data_x[indices], data_y[indices])
            self.learners.append(learner_instance)

            
        if self.verbose:
            print(f"Trained {len(self.learners)} learners using bagging.")


    def query(self, points):
        return np.mean([learner.query(points) for learner in self.learners], axis=0)