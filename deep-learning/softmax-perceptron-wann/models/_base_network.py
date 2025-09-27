""" 			  		 			     			  	   		   	  			  	
Models Base.  (c) 2021 Georgia Tech

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

class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):
        self.input_size = input_size
        self.num_classes = num_classes
        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        """
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        """
        scores -= np.max(scores, axis=1, keepdims=True)  # Stability fix
        exp_scores = np.exp(scores)
        prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        prob = np.clip(prob, 1e-9, 1.0)  # Prevent numerical instability
        return prob

    def cross_entropy_loss(self, x_pred, y):
        """
        Compute Cross-Entropy Loss based on prediction of the network and labels

        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        """
        N = x_pred.shape[0]
        log_likelihood = -np.log(np.clip(x_pred[np.arange(N), y], 1e-9, 1.0))
        loss = np.sum(log_likelihood) / N
        return loss


    def compute_accuracy(self, x_pred, y):
        """
        Compute the accuracy of the current batch

        :param x_pred: Probabilities from the model (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        """
        predictions = np.argmax(x_pred, axis=1)
        acc = np.mean(predictions == y)
        return acc

    def sigmoid(self, X):
        """
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, layer size)
        """
        return 1 / (1 + np.exp(-X))

    def sigmoid_dev(self, x):
        """
        The analytical derivative of the sigmoid function at x

        :param x: Input data
        :return: The derivative of the sigmoid function at x
        """
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def ReLU(self, X):
        """
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the ReLU activation is applied to the input (N, layer size)
        """
        return np.maximum(0, X)

    def ReLU_dev(self, X):
        """
        Compute the gradient of ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: gradient of ReLU given input X
        """
        return (X > 0).astype(float)