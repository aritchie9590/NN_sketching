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


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def read_data(loss_type, nb_classes=None, seed=None):
    train_imgs = read_idx('train-images.idx3-ubyte')
    train_lab = read_idx('train-labels.idx1-ubyte')

    test_imgs = read_idx('t10k-images.idx3-ubyte')
    test_lab = read_idx('t10k-labels.idx1-ubyte')

    X = train_imgs.reshape(train_imgs.shape[0], -1)

    if (seed is not None):
        np.random.seed(0)

    idx = np.argsort(train_lab)
    X = X[idx]
    y_sorted = train_lab[idx]
    # y_sorted = y_sorted[y_sorted < 2]           # take only 0 and 1 digits
    X = X[:len(y_sorted), :]
    if (loss_type is 'l2'):
        y = (y_sorted < 1).astype(float)
    else:
        y = y_sorted

    indices = np.random.permutation(X.shape[0])
    # indices = np.arange(0, X.shape[0])
    num_train = np.round(0.8 * len(y_sorted)).astype(int)
    training_idx, test_idx = indices[:num_train], indices[num_train:]
    X_train, X_val = X[training_idx, :], X[test_idx, :]
    y_train, y_val = y[training_idx], y[test_idx]

    X_train = X_train.astype(float)
    y_train = y_train.astype(float)

    X_train = X_train / np.max(X_train)
    X_val = X_val / np.max(X_val)

    if (loss_type is 'softmax'):
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


def readCTdata(seed=None):
    ct_data = np.load('/Users/kgilman/Desktop/Deep-Sketch/NN_sketching/sketched_GN_networks/ct_data.npy').item(0)
    data = ct_data['data'].astype(float)
    labels = ct_data['labels'].astype(float)

    np.random.seed(seed)
    n = len(labels)
    train_samp = np.random.permutation(n)
    train_idx = int(0.8 * n)

    X_train = data[train_samp[:train_idx], :]
    y_train = labels[train_samp[:train_idx]]

    X_val = data[train_samp[train_idx:], :]
    y_val = labels[train_samp[train_idx:]]

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
    return ap.maximum(0, z)


def softmax(x):
    exps = ap.exp(x)
    sum_exps = ap.sum(exps, axis=1, keepdims=True)
    S = exps / sum_exps
    return S


def init_weights(params, seed=None):
    if (seed is not None):
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

        if (count < len(params) - 1):
            a = sigmoid(z)
            # a = relu(z)
            # a = z
        else:  # Last layer activation
            if (loss_type is 'l2'):
                # a = sigmoid(z).squeeze()
                # a = relu(z).squeeze()
                a = z.squeeze()
            else:
                a = z
        count += 1

    return a


def make_model(input_dim, output_dim, loss_type, hidden_dim=None, seed=None):
    if (hidden_dim is None):
        # Define one-layer network
        params = [[input_dim, output_dim]]
    else:
        # Define two-layer network parameters
        params = [[input_dim, hidden_dim], [hidden_dim, output_dim]]
        # params = [[input_dim, 100], [100, hidden_dim], [hidden_dim, 10], [10, output_dim]]
        # params = [[input_dim, 100], [100,100], [100,50],[50,50], [50, 25], [25, 25], [25, 10], [10,output_dim]]

    model = lambda w, X: nn(w, X, params, loss_type)
    w_0 = init_weights(params, seed)
    return model, w_0


### Gauss-Newton Solver ###

# loss_type = 'l2'
loss_type = 'softmax'

# data = read_data(loss_type,seed=0)                    #MNIST binary regression
# data = readCTdata(seed=0)                               #CT regression
data = read_data(loss_type, nb_classes=10, seed=0)    #MNIST 10 digit classification CE

sketch_size = 1000
# sketch_size = 500

input_dim = data['X_train'].shape[1]
hidden_dim = None  ## Use for regression
hidden_dim = 200

if (loss_type is 'l2'):
    output_dim = 1
else:
    output_dim = data['y_train'].shape[1]

model, w0 = make_model(input_dim, output_dim, loss_type, hidden_dim, seed=0)

ITER_GN = 5
ITER_GNHS = 20
ITER_GNS = 20
ITER_SGD = 1000
ITER_ADAM = 500
ITER_SGD = 1000
# n_cg_iter = 1000


# lam = 10 / data['X_train'].shape[0]  ## use for l2 regression
lam = 100 / data['X_train'].shape[0]  ## use for MNIST softmax

import gn
import gn_sketch
import gn_half_sketch
import gd

# print("Gauss-Newton...")
# n_cg_iter = 100
# # # Gauss-Newton
# w_star_gn, loss_log_gn, val_err_gn, t_solve_gn = gn.solver(data, model, lam, w0, loss_type, ITER_GN, n_cg_iter, backtrack =
# False)

print("Gauss-Newton Sketch...")
n_cg_iter = 1000
# Gauss-Newton Sketch
w_star_gns, loss_log_sketch, val_err_sketch, t_solve_sketch = gn_sketch.solver(data, model, lam, w0, loss_type,
                                                                               ITER_GNS, n_cg_iter, backtrack=True,
                                                                               sketch_size=sketch_size)

print("Gauss-Newton Half Sketch...")
# Gauss-Newton Half-Sketch (Iterative)
n_cg_iter = 100
w_star_gnhs, loss_log_hsketch, val_err_hsketch, t_solve_hsketch = gn_half_sketch.solver(data, model, lam, w0, loss_type,
                                                                                        ITER_GNHS, n_cg_iter,
                                                                                        backtrack=False,
                                                                                        sketch_size=sketch_size)

# print("SGD Backtrack...")
# # SGD
# w_star_sgd, loss_log_back, val_err_back, t_solve_back = gd.solver(data, model, lam, w0, loss_type, ITER_SGD,
#                                                                     stochastic=True, step='backtrack',
#                                                                   sketch_size=sketch_size)

print("SGD ADAM...")
w_star_sgd, loss_log_adam, val_err_adam, t_solve_adam = gd.solver(data, model, lam, w0, loss_type, ITER_ADAM,
                                                                    stochastic=True, step='adam', lr = 0.01,
                                                                  decay = 1,
                                                                  sketch_size=sketch_size)


print("SGD...")
w_star_sgd, loss_log_const, val_err_const, t_solve_const = gd.solver(data, model, lam, w0, loss_type, ITER_SGD,
                                                                    stochastic=True, step='constant', lr = 1, decay = 1,
                                                                  sketch_size=sketch_size)

plt.subplot(2, 1, 1)
# plt.semilogy(np.arange(len(loss_log_gn)), loss_log_gn, 'k',
plt.semilogy(
    np.arange(len(loss_log_sketch)), loss_log_sketch, 'g',
    np.arange(len(loss_log_hsketch)), loss_log_hsketch,'r',
    # np.arange(len(loss_log_back)), loss_log_back, 'b',
    np.arange(len(loss_log_adam)), loss_log_adam, 'c',
    np.arange(len(loss_log_const)), loss_log_const, 'm'
)
plt.title('Training loss')
plt.grid(True, which='both')
# plt.legend(['Gauss-Newton', 'Gauss-Newton-Sketch','Gauss-Newton Half-Sketch','SGD'])
# plt.legend(['Gauss-Newton-Sketch','Gauss-Newton Half-Sketch','SGD'])

plt.subplot(2, 1, 2)
# plt.semilogy(np.arange(len(val_err_gn)), val_err_gn, 'k',
plt.semilogy(
    np.arange(len(val_err_sketch)), val_err_sketch, 'g',
    np.arange(len(val_err_hsketch)), val_err_hsketch, 'r',
    # np.arange(len(val_err_back)), val_err_back, 'b',
    np.arange(len(val_err_adam)), val_err_adam, 'c',
    np.arange(len(val_err_const)), val_err_const, 'm'
)
plt.xlabel('Iteration')
plt.title('Validation Loss')
plt.grid(True, which='both')
# plt.legend(['Gauss-Newton Half-Sketch','ADAM'])
# plt.legend(['Gauss-Newton-Sketch','Gauss-Newton Half-Sketch','ADAM'])
plt.legend(['Gauss-Newton-Sketch', 'Gauss-Newton Half-Sketch','ADAM','Constant'])
# plt.legend(['Gauss-Newton-Sketch', 'Gauss-Newton Half-Sketch', 'SGD-Backtrack','ADAM','Diminish'])
# plt.legend(['Gauss-Newton','Gauss-Newton-Sketch', 'Gauss-Newton Half-Sketch', 'SGD-Backtrack','ADAM','Diminish'])
plt.show()

plt.subplot(2, 1, 1)
# plt.semilogy(np.arange(len(loss_log_gn)) * t_solve_gn / ITER_GN, loss_log_gn, 'k',
plt.semilogy(
    np.arange(len(loss_log_sketch)) * t_solve_sketch / ITER_GNS, loss_log_sketch, 'g',
    np.arange(len(loss_log_hsketch)) * t_solve_hsketch / ITER_GNHS, loss_log_hsketch, 'r',
    # np.arange(len(loss_log_back)) * t_solve_back / ITER_SGD, loss_log_back, 'b',
    np.arange(len(loss_log_adam)) * t_solve_adam / ITER_ADAM, loss_log_adam, 'c',
    np.arange(len(loss_log_const)) * t_solve_const / ITER_SGD, loss_log_const, 'm'
)
plt.title('Training loss')
plt.grid(True, which='both')
# plt.legend(['Gauss-Newton Half-Sketch','ADAM'])
# plt.legend(['Gauss-Newton-Sketch','Gauss-Newton Half-Sketch','ADAM'])
# plt.legend(['Gauss-Newton-Sketch', 'Gauss-Newton Half-Sketch', 'SGD-Backtrack','ADAM','Diminish'])
# plt.legend(['Gauss-Newton','Gauss-Newton-Sketch', 'Gauss-Newton Half-Sketch', 'SGD-Backtrack','ADAM','Diminish'])

plt.subplot(2, 1, 2)
# plt.semilogy(np.arange(len(val_err_gn)) * t_solve_gn / ITER_GN, val_err_gn, 'k',
plt.semilogy(
    np.arange(len(val_err_sketch)) * t_solve_sketch / ITER_GNS, val_err_sketch, 'g',
    np.arange(len(val_err_hsketch)) * t_solve_hsketch / ITER_GNHS, val_err_hsketch, 'r',
    # np.arange(len(val_err_back)) * t_solve_back / ITER_SGD, val_err_back, 'b',
    np.arange(len(val_err_adam)) * t_solve_adam / ITER_ADAM, val_err_adam, 'c',
    np.arange(len(val_err_const)) * t_solve_const / ITER_SGD, val_err_const, 'm'
)
plt.xlabel('Seconds')
plt.title('Validation loss')
plt.grid(True, which='both')
# plt.legend(['Gauss-Newton Half-Sketch','ADAM'])
# plt.legend(['Gauss-Newton-Sketch','Gauss-Newton Half-Sketch','ADAM'])
plt.legend(['Gauss-Newton-Sketch', 'Gauss-Newton Half-Sketch','ADAM','Constant'])
# plt.legend(['Gauss-Newton-Sketch', 'Gauss-Newton Half-Sketch', 'SGD-Backtrack','ADAM','Diminish'])
# plt.legend(['Gauss-Newton','Gauss-Newton-Sketch', 'Gauss-Newton Half-Sketch', 'SGD-Backtrack','ADAM','Diminish'])
plt.show()

print("I'm done")
