import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_filters = num_filters
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    HH = filter_size
    WW = filter_size
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, HH, WW)
    self.params['b1'] = np.zeros(num_filters)
    self.out_conv_shape = int(H - HH + 1)
    self.out_pool_shape = int(1 + (self.out_conv_shape - 2) / 2)
    self.params['W2'] = weight_scale * np.random.randn(num_filters*(self.out_pool_shape ** 2),hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim,num_classes)
    self.params['b3'] = np.zeros(num_classes) #### need this??
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #t1 = time.time()
    out_conv, cache_conv = conv_forward(X, W1)
    out_relu, cache_relu = relu_forward(out_conv)
    #print(out_relu.shape)
    out_pool, cache_pool = max_pool_forward(out_relu, pool_param)
    #print(out_pool.shape)
    pool_shape = out_pool.shape
    temp = out_pool.reshape(pool_shape[0], -1)
    out_fc1, cache_fc1 = fc_forward(temp, W2, b2)
    # batch norm / dropout here
    out_fc2, cache_fc2 = fc_forward(out_fc1, W3, b3)
    scores = (np.exp(out_fc2).T / np.sum(np.exp(out_fc2), axis=1)).T
    #print('forward time:', time.time() - t1)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    #t2 = time.time()
    loss, dscores = softmax_loss(out_fc2, y)
    _, dW3, db3 = fc_backward(dscores, cache_fc2)
    #batch norm / dropout here
    _, dW2, db2 = fc_backward(dscores @ W3.T, cache_fc1)
    dH1 = (dscores @ W3.T @ W2.T).reshape(X.shape[0], self.num_filters, self.out_pool_shape, self.out_pool_shape)
    d_pool = max_pool_backward(dH1, cache_pool)
    d_relu = relu_backward(d_pool, cache_relu)
    _, dW1 = conv_backward(d_relu, cache_conv)
    #print('backward time:', time.time() - t2)
    grads['b1'] = np.zeros(b1.shape)
    grads['W1'] = dW1
    grads['b2'] = db2
    grads['W2'] = dW2
    grads['b3'] = db3
    grads['W3'] = dW3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
