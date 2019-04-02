import time
import struct
import numpy as np
from autograd import make_vjp
from autograd import make_jvp
from autograd import make_ggnvp
import matplotlib.pyplot as plt
import autograd.numpy as ap

from autograd import grad
from autograd import elementwise_grad

import numpy as np
#import fjlt
from scipy.optimize import minimize

import nn_cg


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def solver(data, f, lam, w0, loss_type, ITERNEWTON, n_cgiter=None, GN=True, iterative = False, backtrack=True,
           sketch_size=None):

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']


    max_backtrack = 50
    backtrack_alpha = 0.1
    backtrack_beta = 0.9
    NTOL = 1e-8

    w = w0.copy()

    loss_log = np.zeros(ITERNEWTON + 1)
    val_err = np.zeros(ITERNEWTON + 1)

    n = X_train.shape[0]

    if(loss_type is 'l2'):
        loss_log[0] = 0.5*np.linalg.norm(f(w, X_train) - y_train)**2 / n +  lam * np.linalg.norm(w) ** 2 / 2
        val_err[0] = np.sum(abs(np.round(f(w,X_val)) - y_val)) / len(y_val)
    else: #softmax
        loss_log[0] = 0.5 * np.linalg.norm(softmax(f(w, X_train)) - y_train) ** 2 / n + lam * np.linalg.norm(
        w) ** 2 / 2  # back propagate
        val_err[0] = np.sum(abs(np.argmax(softmax(f(w, X_val)), axis=1) - np.argmax(y_val, axis=1)) > 0,
                            axis=0) / y_val.shape[0]

    randperm = lambda n: np.random.permutation(n)
    randp = np.hstack((randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), \
            randperm(n),
             randperm(n), randperm(n), randperm(n), randperm(n)))

    start = time.time()
    for iter in range(0, ITERNEWTON):

        # Generate sketched Gauss-Newton-vector product function
        if (sketch_size is None):
            sketch_size = X_train.shape[0]

        # sample_idx = np.random.permutation(n)[:sketch_size]
        sample_idx = randp[sketch_size * (iter) + 1: sketch_size * (iter + 1) + 1]
        Xs = X_train[sample_idx, :]

        if (GN): #Gauss-Newton algorithms

            if(iterative): #Iterative GN sketch
                vjp = make_vjp(f)(w, X_train)[0]

                if(loss_type is 'l2'): #Regression
                    e = f(w, X_train) - y_train
                else:   #softmax
                    e = softmax(f(w, X_train)) - y_train

            else:   #Randomized GN sketch
                vjp = make_vjp(f)(w, Xs)[0]
                if (loss_type is 'l2'): #Regression
                    e = f(w, Xs) - y_train[sample_idx]
                else:    #Softmax
                    e = softmax(f(w, Xs)) - y_train[sample_idx]

            vjp_s = make_vjp(f)(w, Xs)[0]

            loss = 0.5 * np.linalg.norm(e) ** 2 / n  # compute loss

            grads = vjp(e) / n + lam * w  # compute gradient
            # ggnvp = make_ggnvp(f)(w, Xs, params)
            # ggnvp_fxn = lambda v: ggnvp(v) / n + lam * v  # Make Gauss-Newton-vector product function

            jvp_sketch = make_jvp(f)(w, Xs)
            ggnvp_fxn = lambda v: vjp_s(jvp_sketch(v)[1]) / n + lam * v

            # Solve CG for the Newton direction
            dw, cost_log = nn_cg.nn_cgm(ggnvp_fxn, grads, np.zeros(len(grads)), n_cgiter, True)

        else:  # gradient descent
            vjp = make_vjp(f)(w, Xs)[0]
            if(loss_type is 'l2'):
                e = f(w, Xs) - y_train[sample_idx]
            else:
                e = softmax(f(w, Xs)) - y_train[sample_idx]

            loss = 0.5 * np.linalg.norm(e) ** 2 / n  # compute loss

            grads = vjp(e) / n + lam * w  # compute gradient
            dw = grads.copy()

        # Perform backtracking line-search
        val = loss + lam * np.linalg.norm(w) ** 2 / 2
        fprime = grads.T @ dw

        t = 1

        if (backtrack):
            bts = 0
            alpha = backtrack_alpha
            beta = backtrack_beta

            if(loss_type is 'l2'):
                fxn_val = lambda v: 0.5*np.linalg.norm(f(w - t*dw,X_train) - y_train)**2/n + lam * np.linalg.norm(w -
                                                                                                                 t * dw) ** 2 / 2
            else:
                fxn_val = lambda v: 0.5 * np.linalg.norm(
                    softmax(f(w - t * dw, X_train)) - y_train) ** 2 / n + lam * np.linalg.norm(w - t * dw) ** 2 / 2

            while (fxn_val(w - t*dw) > val + alpha * t * fprime):
                t = beta * t
                bts += 1
                if bts > max_backtrack:
                    print("Maximum backtracking reached, accuracy not guaranteed")
                    break
        else:  # Use decreasing step size
            if(GN):
                t = 1 / (iter + 1)
            else:
                t = 3000 / (iter + 1)

        # Update the NN params
        w = w - t * dw

        loss_log[iter+1] = val

        if(loss_type is 'l2'):
            val_err[iter+1] = np.sum(abs(np.round(f(w, X_val)) - y_val)) / len(y_val)
        else:
            val_err[iter + 1] = np.sum(
                abs(np.argmax(softmax(f(w, X_val)), axis=1) - np.argmax(y_val, axis=1)) > 0,
                axis=0) / y_val.shape[0]

        print(val)

        if abs(fprime) < NTOL:
            loss_log = loss_log[:iter]
            break

    end = time.time()
    t_solve = end - start
    return w, loss_log, val_err, t_solve


def read_data(loss_type,seed=None):

    train_imgs = read_idx('train-images.idx3-ubyte')
    train_lab = read_idx('train-labels.idx1-ubyte')

    test_imgs = read_idx('t10k-images.idx3-ubyte')
    test_lab = read_idx('t10k-labels.idx1-ubyte')

    X = train_imgs.reshape(train_imgs.shape[0], -1)

    if(seed is not None):
        np.random.seed(0)

    idx = np.argsort(train_lab)
    X = X[idx]
    y_sorted = train_lab[idx]

    if(loss_type is 'l2'):
        y = (y_sorted < 1).astype(float)
    else:
        y = y_sorted

    indices = np.random.permutation(X.shape[0])
    # indices = np.arange(0, X.shape[0])
    training_idx, test_idx = indices[:50000], indices[50000:]
    X_train, X_val = X[training_idx, :], X[test_idx, :]
    y_train, y_val = y[training_idx], y[test_idx]

    X_train = X_train.astype(float)
    y_train = y_train.astype(float)

    X_train = X_train / np.max(X_train)
    X_val = X_val / np.max(X_val)

    if(loss_type is 'softmax'):
        nb_classes = 10
        targets = y_train.astype(int)
        y_train = np.eye(nb_classes)[targets]

        targets = y_val.astype(int)
        y_val = np.eye(nb_classes)[targets]


    data = {
        'X_train': X_train,  # training data
        'y_train': y_train,  # training labels
        'X_val': X_val,  # validation data
        'y_val': y_val  # validation labels
    }
    return data


def fc_forward(x, w, b=None):
    if (b is None):
        b = 0
    return x @ w + b

def sigmoid(z):
    return 1 / (1 + ap.exp(-z))

def relu(z):
    return ap.maximum(0.0, z)

def softmax(x):
  exps = ap.exp(x)
  sum_exps = ap.sum(exps,axis=1,keepdims=True)
  S = exps / sum_exps
  return S

def init_weights(params,seed=None):
    if(seed is not None):
        ap.random.seed(seed)
    d = 0
    for each in params:
        din, dout = each
        d += din * dout + dout

    return ap.random.randn(d)  # initialize with random weights


def nn(w, X, params, loss_type):
    idx = 0
    a = X.copy()
    count = 0
    for each in params:
        din, dout = each
        inc = int(din * dout)
        W = w[idx:idx + inc].reshape(din, dout)
        idx += inc
        b = w[idx:idx + dout]
        idx += dout

        z = fc_forward(a, W, b)

        if(count < len(params)-1):
            a = sigmoid(z)
            # a = relu(z)
        else:
            if(loss_type is 'l2'):
                a = sigmoid(z).squeeze()
            else:
                a = z
        count +=1

    return a

def make_model(input_dim,output_dim,loss_type,hidden_dim = None,seed=None):
    if(hidden_dim is None):
        # Define one-layer network
        params = [[input_dim, output_dim]]
    else:
        # Define two-layer network parameters
        params = [[input_dim, hidden_dim],[hidden_dim, output_dim]]

    model = lambda w,X: nn(w, X, params,loss_type)
    w_0 = init_weights(params,seed)
    return model, w_0


### Gauss-Newton Solver ###

loss_type = 'softmax'
data = read_data(loss_type,seed=0)
sketch_size = 784

input_dim = data['X_train'].shape[1]
hidden_dim = 15

if(loss_type is 'l2'):
    output_dim = 1
else:
    output_dim = data['y_train'].shape[1]

model,w0 = make_model(input_dim,output_dim,loss_type,hidden_dim,seed=0)

ITER_GN = 5
ITER_GNS = 10
ITER_SGD = 50
n_cg_iter = 5

lam = 0.0001
# lam = 10 / data['X_train'].shape[0] #use for l2 regression

# Gauss-Newton
# w_star_gn, loss_log, t_solve = solver(data, model, lam, w0, ITER_GN, n_cg_iter)

# Gauss-Newton Sketch
w_star_gns, loss_log_sketch, val_err_sketch, t_solve_sketch = solver(data, model, lam, w0, loss_type,
                                                                     ITER_GNS, n_cg_iter,
                                                     GN=True, iterative = False,
                                                     backtrack=False, sketch_size=sketch_size)

# test = softmax(f2(w_star_gns,X_val,params))
# test_labels = np.argmax(test,axis=1)
# plt.subplot(2,1,1)
# plt.hist(test_labels)
# plt.subplot(2,1,2)
# plt.hist(np.argmax(y_val,axis=1))
# plt.show()

# SGD
w_star_sgd, loss_log_sgd, val_err_sgd, t_solve_sgd = solver(data, model, lam, w0, loss_type,ITER_SGD,
                                                            GN=False,
                                               backtrack=False, sketch_size=sketch_size)


# plt.semilogy(np.arange(len(loss_log)), loss_log, 'r', np.arange(len(loss_log_sketch)), loss_log_sketch, 'g',
#              np.arange(len(loss_log_sgd)), loss_log_sgd, 'b')
# plt.xlabel('Iteration')
# plt.title('Cost function')
# plt.grid(True, which='both')
# plt.legend(['Gauss-Newton', 'Gauss-Newton-Sketch', 'SGD'])

plt.subplot(2,1,1)
plt.semilogy(np.arange(len(loss_log_sketch)), loss_log_sketch, 'g',np.arange(len(loss_log_sgd)), loss_log_sgd, 'b')
plt.xlabel('Iteration')
plt.title('Training loss')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton-Sketch', 'SGD'])

plt.subplot(2,1,2)
plt.plot(np.arange(len(val_err_sketch)), val_err_sketch, 'g',np.arange(len(val_err_sgd)), val_err_sgd, 'b')
plt.xlabel('Iteration')
plt.title('Validation Loss')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton-Sketch', 'SGD'])
plt.show()

# plt.subplot(1, 2, 1)
# plt.semilogy(np.arange(len(loss_log)) * t_solve / ITER_GN, loss_log, 'r',
#              np.arange(len(loss_log_sketch)) * t_solve_sketch / ITER_GNS, loss_log_sketch, 'g',
#              np.arange(len(loss_log_sgd)) * t_solve_sgd / ITER_SGD, loss_log_sgd, 'b')
# plt.xlabel('Seconds')
# plt.title('Cost function')
# plt.grid(True, which='both')
# plt.legend(['Gauss-Newton', 'Gauss-Newton-Sketch', 'SGD'])

plt.subplot(2,1,1)
plt.semilogy(np.arange(len(loss_log_sketch)) * t_solve_sketch / ITER_GNS, loss_log_sketch, 'g', np.arange(len(
    loss_log_sgd)) * t_solve_sgd / ITER_SGD, loss_log_sgd, 'b')
plt.xlabel('Seconds')
plt.title('Training loss')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton-Sketch', 'SGD'])


plt.subplot(2,1,2)
plt.plot(np.arange(len(val_err_sketch)) * t_solve_sketch / ITER_GNS, val_err_sketch, 'g', np.arange(len(
    val_err_sgd)) * t_solve_sgd / ITER_SGD, val_err_sgd, 'b')
plt.xlabel('Seconds')
plt.title('Validation loss')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton-Sketch', 'SGD'])
plt.show()



print("I'm done")
