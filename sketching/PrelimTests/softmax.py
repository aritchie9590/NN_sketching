import numpy as np
from layers import *


class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################
        self.twoLayer = True
        if hidden_dim is None:
            self.twoLayer = False
    
        if hidden_dim is None:
            self.params['W1'] = weight_scale * np.random.randn(input_dim,num_classes)
            self.params['b1'] = np.zeros(num_classes)
        else:
            self.params['W2'] = weight_scale * np.random.randn(hidden_dim,num_classes)
            self.params['b2'] = 0.0
            self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
            self.params['b1'] = np.zeros(hidden_dim)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1 = self.params['W1']
        b1 = self.params['b1']
        if 'W2' in self.params:
            W2 = self.params['W2']
            b2 = self.params['b2']
    
        if not self.twoLayer:
            result, cache_fc = fc_forward(X, W1, b1)
            scores = (np.exp(result).T / np.sum(np.exp(result), axis=1)).T
            #scores = np.argmax(scores, axis=1)
            # for 0,1 loss
        else:
            result_fc1, cache_fc1 = fc_forward(X, W1, b1)
            result_relu, cache_relu = relu_forward(result_fc1)
            result_fc2, cache_fc2 = fc_forward(result_relu, W2, b2)
            scores = (np.exp(result_fc2).T / np.sum(np.exp(result_fc2), axis=1)).T
            #scores = np.argmax(scores, axis=1)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        if not self.twoLayer:
            loss, dscores = softmax_loss(result, y)
            _, dW1, db1 = fc_backward(dscores, cache_fc)
            grads['b1'] = db1
            grads['W1'] = dW1
        else:
            loss, dscores = softmax_loss(result_fc2, y)
            _, dW2, db2 = fc_backward(dscores, cache_fc2)
            d_relu = relu_backward(dscores @ W2.T, cache_relu)
            _, dW1, db1 = fc_backward(d_relu, cache_fc1)
            grads['b2'] = db2
            grads['W2'] = dW2
            grads['b1'] = db1
            grads['W1'] = dW1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
