""" 			  		 			     			  	   		   	  			  	
Softmax Regression Model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

# Do not use packages that are not in standard distribution of python
import numpy as np
from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        Initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, num_classes)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of images (N, 28x28)
        :param y: labels of images in the batch (N,)
        :param mode: training mode - if 'train', compute gradients; otherwise, return loss and accuracy.
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        # Forward pass: Linear transformation
        logits = np.dot(X, self.weights['W1'])  # (N, num_classes)
        
        # Apply ReLU activation
        relu_output = self.ReLU(logits)

        # Softmax output
        probs = self.softmax(relu_output)

        # Compute cross-entropy loss
        loss = self.cross_entropy_loss(probs, y)

        # Compute accuracy
        accuracy = self.compute_accuracy(probs, y)

        if mode != 'train':
            return loss, accuracy

        # Backward pass
        # Compute softmax gradient
        dL_drelu = probs
        dL_drelu[np.arange(len(y)), y] -= 1  # One-hot encoding gradient adjustment
        dL_drelu /= len(y)

        # Compute gradients for W1
        dL_dlogits = dL_drelu * self.ReLU_dev(logits)  # Apply ReLU derivative
        self.gradients['W1'] = np.dot(X.T, dL_dlogits)  # Weight gradient update

        return loss, accuracy