import numpy as np

from layers import *

class LogisticClassifier(object):
  """
  A logistic regression model with optional hidden layers.
  
  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    ############################################################################
    # TODO: Initialize the weights and biases of the model. Weights            #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases (if any) using the keys 'W2' and 'b2'.                        #
    ############################################################################
    self.twoLayer = True
    if hidden_dim is None:
        self.twoLayer = False
    
    if hidden_dim is None:
        self.params['W1'] = weight_scale * np.random.randn(input_dim,1)
        self.params['b1'] = 0.0
    else:
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim,1)
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
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N,) where scores[i] represents the logit for X[i]
    
    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  

    W1 = self.params['W1']
    b1 = self.params['b1']
    if 'W2' in self.params:
        W2 = self.params['W2']
        b2 = self.params['b2']

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the model, computing the            #
    # scores for X and storing them in the scores variable.                    #
    ############################################################################
    
    if not self.twoLayer:
        result, cache_fc = fc_forward(X, W1, b1)
        scores = 1 / (1 + np.exp(-result))
        scores = (scores > 0.5).astype(int)
    else:
        result_fc1, cache_fc1 = fc_forward(X, W1, b1)
        result_relu, cache_relu = relu_forward(result_fc1)
        result_fc2, cache_fc2 = fc_forward(result_relu, W2, b2)
        scores = 1 / (1 + np.exp(-result_fc2))
        scores = (result_fc2 > 0.5).astype(int) # for 0,1 loss
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores.flatten()
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the model. Store the loss          #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss and make sure that grads[k] holds the gradients for self.params[k]. #
    # Don't forget to add L2 regularization.                                   #
    #                                                                          #
    ############################################################################
    if not self.twoLayer:
        loss, dscores = logistic_loss(scores, y)
        _, dW1, db1 = fc_backward(dscores.reshape(dscores.size,1), cache_fc)
        grads['b1'] = db1
        grads['W1'] = dW1
    else:
        loss, dscores = logistic_loss(scores, y)
        _, dW2, db2 = fc_backward(dscores.reshape(dscores.size,1), cache_fc2)
        d_relu = relu_backward((W2 * dscores).T, cache_relu)
        _, dW1, db1 = fc_backward(d_relu, cache_fc1)
        grads['b2'] = db2
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['W1'] = dW1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    
    return loss, grads
