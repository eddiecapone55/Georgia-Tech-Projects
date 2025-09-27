""" 			 	     			  	   		   	  			  	
MLP Model.  (c) 2021 Georgia Tech

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

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)
        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):
        """
        Initialize weights of the network.
        """
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        self.weights['b1'] = np.zeros(self.hidden_size)

        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)
        self.weights['b2'] = np.zeros(self.num_classes)

        # Initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        """
        Forward and backward pass of the two-layer neural network.
        """
        # Forward pass: Layer 1 (Linear -> Sigmoid Activation)
        z1 = np.dot(X, self.weights['W1']) + self.weights['b1'].reshape(1, -1)
        a1 = self.sigmoid(z1)  # Apply sigmoid activation

        # Forward pass: Layer 2 (Linear -> Softmax)
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2'].reshape(1, -1)
        probs = self.softmax(z2)  # Apply softmax activation

        # Compute loss and accuracy
        loss = self.cross_entropy_loss(probs, y)
        accuracy = self.compute_accuracy(probs, y)

        if mode == 'train':
            N = X.shape[0]

            # Compute gradient of softmax loss
            dL_dz2 = probs.copy()
            dL_dz2[np.arange(N), y] -= 1
            dL_dz2 /= N  # Normalize by batch size

            # Compute gradients for W2 and b2
            self.gradients['W2'] = np.dot(a1.T, dL_dz2)
            self.gradients['b2'] = np.sum(dL_dz2, axis=0)

            # Backpropagate through first layer using the sigmoid derivative on z1
            dL_da1 = np.dot(dL_dz2, self.weights['W2'].T)
            dL_dz1 = dL_da1 * self.sigmoid_dev(z1)  # Use pre-activation z1 for derivative

            # Compute gradients for W1 and b1
            self.gradients['W1'] = np.dot(X.T, dL_dz1)
            self.gradients['b1'] = np.sum(dL_dz1, axis=0)

        return loss, accuracy