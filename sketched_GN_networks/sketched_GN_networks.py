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

import nn_cg


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


train_imgs = read_idx('train-images.idx3-ubyte')
train_lab = read_idx('train-labels.idx1-ubyte')

test_imgs = read_idx('t10k-images.idx3-ubyte')
test_lab = read_idx('t10k-labels.idx1-ubyte')

X = train_imgs.reshape(train_imgs.shape[0], -1)

idx = np.argsort(train_lab)
X = X[idx]
y_sorted = train_lab[idx]

y = (y_sorted < 1).astype(float)

# indices = np.random.permutation(X.shape[0])
indices = np.arange(0, X.shape[0])
training_idx, test_idx = indices[:60000], indices[60000:]
X_train, X_val = X[training_idx, :], X[test_idx, :]
y_train, y_val = y[training_idx], y[test_idx]

val_idx = np.random.shuffle(y_train)

data = {
    'X_train': X_train,  # training data
    'y_train': y_train,  # training labels
    'X_val': X_val,  # validation data
    'y_val': y_val  # validation labels
}

X_train = X_train.astype(float)
y_train = y_train.astype(float)


def solver(X_train, y_train, f, params, lam, w0, ITERNEWTON, n_cgiter=None, GN=True, backtrack=True, sketch_size=None):
    max_backtrack = 50
    backtrack_alpha = 0.2
    backtrack_beta = 0.5
    NTOL = 1e-8

    w = w0.copy()

    loss_log = np.zeros(ITERNEWTON)
    start = time.time()
    for iter in range(0, ITERNEWTON):

        # Generate sketched Gauss-Newton-vector product function
        if (sketch_size is None):
            sketch_size = X_train.shape[0]

        sample_idx = np.random.permutation(n)[:sketch_size]
        Xs = X_train[sample_idx, :]

        vjp = make_vjp(f)(w, Xs, params)[0]
        e = f(w, Xs, params) - y_train[sample_idx]  # forward propagate

        loss = 0.5 * np.linalg.norm(e) ** 2 / n  # compute loss

        if (GN):
            grad = vjp(e) / n + lam * w  # compute gradient
            ggnvp = make_ggnvp(f)(w, Xs, params)
            ggnvp_fxn = lambda v: ggnvp(v) / n + lam * v  # Make Gauss-Newton-vector product function

            # Solve CG for the Newton direction
            dw, cost_log = nn_cg.nn_cgm(ggnvp_fxn, grad, np.zeros(len(grad)), lam, n_cgiter, True)
        else:  # gradient descent
            grad = vjp(e) / n + lam * w  # compute gradient
            dw = grad.copy()

        # Perform backtracking line-search
        val = loss + lam * np.linalg.norm(w) ** 2 / 2
        fprime = grad.T @ dw

        t = 1

        if (backtrack):
            bts = 0
            alpha = backtrack_alpha
            beta = backtrack_beta
            while (loss + lam * np.linalg.norm(w - t * dw) ** 2 / 2 > val + alpha * t * fprime):
                t = beta * t
                bts += 1
                if bts > max_backtrack:
                    print("Maximum backtracking reached, accuracy not guaranteed")
                    break
        else:  # Use decreasing step size
            if(GN):
                t = 1
            else:
                t = 5000 / (iter + 1)

        # Update the NN params
        w = w - t * dw

        loss_log[iter] = val

        if abs(fprime) < NTOL:
            loss_log = loss_log[:iter]
            break

    end = time.time()
    t_solve = end - start
    return w, loss_log, t_solve


n = X_train.shape[0]
X_train = X_train / np.max(X_train)

lam = 10 / n


# # Define single-layer sigmoid-activated network
# def f(w, X):
#     z = X @ w
#     a = 1 / (1 + ap.exp(-z))
#     return a


def fc_forward(x, w, b=None):
    if (b is None):
        b = 0
    return x @ w + b


def sigmoid(z):
    return 1 / (1 + ap.exp(-z))


def relu(z):
    out = ap.maximum(0.0, z)
    return out


# #Define one-layer network
# input_dim = X.shape[1]
# params = {
#     "L1": [input_dim,1],
# }


# Define two-layer network parameters
input_dim = X.shape[1]
hidden_dim = 200
output_dim = 1

params = {
    "L1": [input_dim, hidden_dim],
    "L2": [hidden_dim, output_dim]
}

def f2(w, X, params):
    idx = 0
    a = X.copy()
    for p, q in params.items():
        din, dout = params[p]
        inc = int(din * dout)
        W = w[idx:idx + inc].reshape(din, dout)
        idx += inc
        b = w[idx:idx + dout]
        idx += dout

        z = fc_forward(a, W, b)
        a = sigmoid(z)

    return a.squeeze()


### Gauss-Newton Solver ###

sketch_size = 784

ITER_GN = 5
ITER_GNS = 10
ITER_SGD = 20
n_cg_iter = 5

d = 0
for p, q in params.items():
    din, dout = params[p]
    d += din * dout + dout

w0 = ap.random.randn(d)  # initialize with random weights

# Gauss-Newton
w_star_gn, loss_log, t_solve = solver(X_train, y_train, f2, params, lam, w0, ITER_GN, n_cg_iter)

# Gauss-Newton Sketch
w_star_gns, loss_log_sketch, t_solve_sketch = solver(X_train, y_train, f2, params, lam, w0, ITER_GNS, n_cg_iter,
                                                     GN=True,
                                                     backtrack=True, sketch_size=sketch_size)

# SGD
w_star_sgd, loss_log_sgd, t_solve_sgd = solver(X_train, y_train, f2, params, lam, w0, ITER_SGD, GN=False,
                                               backtrack=False, sketch_size=sketch_size)


plt.semilogy(np.arange(len(loss_log)), loss_log, 'r', np.arange(len(loss_log_sketch)), loss_log_sketch, 'g',
             np.arange(len(loss_log_sgd)), loss_log_sgd, 'b')
plt.xlabel('Iteration')
plt.title('Cost function')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton', 'Gauss-Newton-Sketch', 'SGD'])

# plt.subplot(1, 2, 2)
# plt.semilogy(np.arange(len(loss_log_sketch)), loss_log_sketch, 'g',np.arange(len(loss_log_sgd)), loss_log_sgd, 'b')
# plt.xlabel('Iteration')
# plt.title('Cost function')
# plt.grid(True, which='both')
# plt.legend(['Gauss-Newton-Sketch', 'SGD'])

plt.show()

plt.subplot(1, 2, 1)
plt.semilogy(np.arange(len(loss_log)) * t_solve / ITER_GN, loss_log, 'r',
             np.arange(len(loss_log_sketch)) * t_solve_sketch / ITER_GNS, loss_log_sketch, 'g',
             np.arange(len(loss_log_sgd)) * t_solve_sgd / ITER_SGD, loss_log_sgd, 'b')
plt.xlabel('Seconds')
plt.title('Cost function')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton', 'Gauss-Newton-Sketch', 'SGD'])

plt.subplot(1, 2, 2)
plt.semilogy(np.arange(len(loss_log_sketch)) * t_solve_sketch / ITER_GNS, loss_log_sketch, 'g', np.arange(len(
    loss_log_sgd)) * t_solve_sgd / ITER_SGD, loss_log_sgd, 'b')
plt.xlabel('Seconds')
plt.title('Cost function')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton-Sketch', 'SGD'])
plt.show()

print("I'm done")
