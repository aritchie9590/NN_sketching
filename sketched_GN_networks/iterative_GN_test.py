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


def gd_test(data, f, lam, w0, loss_type, ITERNEWTON, n_cgiter=None, GN=True, iterative = False, backtrack=True,
           sketch_size=None):

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    max_backtrack = 50
    backtrack_alpha = 0.25
    backtrack_beta = 0.95
    NTOL = 1e-8

    w = w0.copy()

    w_log = np.zeros((len(w0),ITERNEWTON + 1))
    w_log[:,0] = w
    loss_log = np.zeros(ITERNEWTON + 1)
    val_err = np.zeros(ITERNEWTON + 1)

    n = X_train.shape[0]

    loss_log[0] = 0.5*np.linalg.norm(f(w, X_train) - y_train)**2 / n
    val_err[0] = 0.5*np.linalg.norm(f(w, X_val) - y_val)**2 / n

    randperm = lambda n: np.random.permutation(n)
    randp = np.hstack((randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), \
                       randperm(n),
                       randperm(n), randperm(n), randperm(n), randperm(n)))

    print(np.linalg.norm(f(w, X_train) - y_train) ** 2 / n)
    start = time.time()
    for iter in range(0, ITERNEWTON):

        # sample_idx = randp[sketch_size * (iter) + 1: sketch_size * (iter + 1) + 1]
        # Xs = X_train[sample_idx, :]
        #
        vjp = make_vjp(f)(w, X_train)[0]
        # vjp = make_vjp(f)(w, Xs)[0]
        #
        e = f(w, X_train) - y_train
        # e = f(w, Xs) - y_train[sample_idx]
        #
        # loss = np.linalg.norm(e) ** 2  / n # compute loss
        loss = np.linalg.norm(e) ** 2   # compute loss
        #
        grads = vjp(e) # compute gradient
        # grads = vjp(e) / n

        dw = - grads

        # Perform backtracking line-search
        val = loss
        fprime = grads.T @ dw

        t = 1

        alpha = backtrack_alpha
        beta = backtrack_beta

        while (np.linalg.norm(f(w + t * dw,X_train) - y_train) ** 2 > val + alpha * t * fprime):
        # while (np.linalg.norm(f(w + t * dw, Xs) - y_train[sample_idx]) ** 2 / n > val + alpha * t * fprime):
            t = beta * t

        ## Update the NN params

        w = w + t * dw
        w_log[:,iter + 1] = w


        val_err[iter +1] = 0.5*np.linalg.norm(f(w,X_val) - y_val)**2/n

        loss = np.linalg.norm(f(w, X_train) - y_train) ** 2 / n
        loss_log[iter+1] = loss
        print(loss)

        if np.sum(np.power(dw, 2)) <= NTOL :
            loss_log = loss_log[:iter]
            val_err = val_err[:iter]
            w_log = w_log[:,:iter]
            break

    end = time.time()
    t_solve = end - start
    return w_log, loss_log, val_err, t_solve


def gn_test(data, f, lam, w0, loss_type, ITERNEWTON, n_cgiter=None, GN=True, iterative = False, backtrack=True,
           sketch_size=None):

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    backtrack_alpha = 0.25
    backtrack_beta = 0.95
    NTOL = 1e-8

    w = w0.copy()

    w_log = np.zeros((len(w0),ITERNEWTON + 1))
    w_log[:,0] = w
    loss_log = np.zeros(ITERNEWTON + 1)
    val_err = np.zeros(ITERNEWTON + 1)

    n = X_train.shape[0]

    loss_log[0] = 0.5*np.linalg.norm(f(w, X_train) - y_train)**2 / n
    val_err[0] = 0.5*np.linalg.norm(f(w, X_val) - y_val)**2 / n

    print(np.linalg.norm(f(w, X_train) - y_train) ** 2 / n)
    start = time.time()
    for iter in range(0, ITERNEWTON):

        vjp = make_vjp(f)(w, X_train)[0]
        e = f(w, X_train) - y_train
        loss = np.linalg.norm(e) ** 2   # compute loss

        g = vjp(e)

        jvp = make_jvp(f)(w, X_train)

        subprob = lambda v: g.T @ v + 0.5 * np.sum(np.power(jvp(v)[1], 2))
        subgrad = lambda v: g + vjp(jvp(v)[1])


        res = minimize(subprob, w, jac=subgrad, method="BFGS", \
                       options={'gtol': 1e-3, 'norm': 2.0, 'eps': 0.1, \
                                'maxiter': None, 'disp': False})
        dw = res.x

        decr = np.sum(dw * g)

        # Perform backtracking line-search
        val = loss

        t = 1

        alpha = backtrack_alpha
        beta = backtrack_beta

        while (np.linalg.norm(f(w + t * dw,X_train) - y_train) ** 2 > val + alpha * t * decr):
            t = beta * t

        # Update the NN params
        w = w + t * dw
        w_log[:,iter + 1] = w

        val_err[iter +1] = np.linalg.norm(f(w,X_val) - y_val)**2 / n

        loss = np.linalg.norm(f(w, X_train) - y_train) ** 2 / n
        loss_log[iter + 1] = loss
        print(loss)

        if abs(decr) < NTOL:
            loss_log = loss_log[:iter]
            val_err = val_err[:iter]
            w_log = w_log[:,:iter]
            break

    end = time.time()
    t_solve = end - start
    return w_log, loss_log, val_err, t_solve

def gn_sketch_test(data, f, lam, w0, loss_type, ITERNEWTON, n_cgiter=None, GN=True, iterative = False, backtrack=True,
           sketch_size=None):

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    backtrack_alpha = 0.25
    backtrack_beta = 0.95
    NTOL = 1e-8

    w = w0.copy()

    w_log = np.zeros((len(w0),ITERNEWTON + 1))
    w_log[:,0] = w
    loss_log = np.zeros(ITERNEWTON + 1)
    val_err = np.zeros(ITERNEWTON + 1)

    n = X_train.shape[0]

    loss_log[0] = 0.5*np.linalg.norm(f(w, X_train) - y_train)**2 / n
    val_err[0] = 0.5*np.linalg.norm(f(w, X_val) - y_val)**2 / n

    randperm = lambda n: np.random.permutation(n)
    randp = np.hstack((randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), \
                       randperm(n),
                       randperm(n), randperm(n), randperm(n), randperm(n)))

    start = time.time()
    print(np.linalg.norm(f(w, X_train) - y_train) ** 2 / n)
    for iter in range(0, ITERNEWTON):

        sample_idx = randp[sketch_size * (iter) + 1: sketch_size * (iter + 1) + 1]
        Xs = X_train[sample_idx, :]

        vjp = make_vjp(f)(w, Xs)[0]
        e = f(w, Xs) - y_train[sample_idx]
        loss = np.linalg.norm(e) ** 2 / n   # compute loss

        g = vjp(e)

        jvp = make_jvp(f)(w, Xs)

        subprob = lambda v: (g.T @ v + 0.5 * np.sum(np.power(jvp(v)[1], 2))) / n
        subgrad = lambda v: (g + vjp(jvp(v)[1])) / n
        res = minimize(subprob, w, jac=subgrad, method="CG", \
                       options={'gtol': 1e-3, 'norm': 2.0, 'eps': 0.1, \
                                'maxiter': 10, 'disp': False})
        dw = res.x

        decr = np.sum(dw * g / n)

        # Perform backtracking line-search
        val = loss

        t = 1

        alpha = backtrack_alpha
        beta = backtrack_beta

        while (np.linalg.norm(f(w + t * dw,Xs) - y_train[sample_idx]) ** 2 / n > val + alpha * t * decr):
            t = beta * t

        # Update the NN params
        w = w + t * dw
        w_log[:,iter + 1] = w

        val_err[iter +1] = np.linalg.norm(f(w,X_val) - y_val)**2 / n

        loss = np.linalg.norm(f(w, X_train) - y_train) ** 2 / n
        loss_log[iter + 1] = loss
        print(loss)

        if abs(decr) < NTOL:
            loss_log = loss_log[:iter]
            val_err = val_err[:iter]
            w_log = w_log[:,:iter]
            break

    end = time.time()
    t_solve = end - start
    return w_log, loss_log, val_err, t_solve


def gn_half_sketch_test(data, f, lam, w0, loss_type, ITERNEWTON, n_cgiter=None, GN=True, iterative = False,
                      backtrack=True,
           sketch_size=None):

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    backtrack_alpha = 0.25
    backtrack_beta = 0.95
    NTOL = 1e-8

    w = w0.copy()

    w_log = np.zeros((len(w0),ITERNEWTON + 1))
    w_log[:,0] = w
    loss_log = np.zeros(ITERNEWTON + 1)
    val_err = np.zeros(ITERNEWTON + 1)

    n = X_train.shape[0]

    loss_log[0] = 0.5*np.linalg.norm(f(w, X_train) - y_train)**2 / n
    val_err[0] = 0.5*np.linalg.norm(f(w, X_val) - y_val)**2 / n

    divide_by = 1
    print(np.linalg.norm(f(w, X_train) - y_train) ** 2 / n)
    start = time.time()
    for iter in range(0, ITERNEWTON):


        sample_idx = np.random.choice(n,sketch_size,False)
        Xs = X_train[sample_idx, :]

        vjp = make_vjp(f)(w, X_train)[0]
        e = f(w, X_train) - y_train
        loss = np.linalg.norm(e) ** 2 / divide_by   # compute loss

        g = vjp(e)

        # vjp_sk = make_vjp(f)(w, Xs)[0]          ### This is computationally expensive and wasteful
        jvp = make_jvp(f)(w, Xs)

        subprob = lambda v: (g.T @ v + 0.5 * np.sum(np.power(jvp(v)[1], 2))) / divide_by
        # subprob2 = lambda v: (g.T @ v + 0.5 * np.sum(np.power(Xs @ v, 2))) / divide_by

        def testfxn(v):
            pad = np.zeros(n)
            pad[sample_idx] = jvp(v)[1]
            return vjp(pad)

        # subgrad = lambda v: (g + vjp_sk(jvp(v)[1])) / divide_by
        # subgrad2 = lambda v: (g + Xs.T @ (Xs @ v)) / divide_by
        subgrad = lambda v: (g + testfxn(v)) / divide_by

        res = minimize(subprob, w, jac=subgrad, method="CG", \
                       options={'gtol': 1e-3, 'norm': 2.0, 'eps': 0.1, \
                                'maxiter': 10, 'disp': False})
        dw = res.x

        decr = np.sum(dw * g / divide_by)

        # Perform backtracking line-search
        val = loss

        # t = 0.2
        t = 1

        alpha = backtrack_alpha
        beta = backtrack_beta

        while (np.linalg.norm(f(w + t * dw,X_train) - y_train) ** 2 / divide_by > val + alpha * t * decr):
            t = beta * t

        # Update the NN params
        w = w + t * dw
        w_log[:,iter + 1] = w

        val_err[iter +1] = np.linalg.norm(f(w,X_val) - y_val)**2 / n

        loss = np.linalg.norm(f(w, X_train) - y_train) ** 2 / n
        loss_log[iter + 1] = loss
        print(loss)

        if abs(decr) < NTOL:
            loss_log = loss_log[:iter]
            val_err = val_err[:iter]
            w_log = w_log[:,:iter]
            break

    end = time.time()
    t_solve = end - start
    return w_log, loss_log, val_err, t_solve


def fc_forward(x, w, b=None):
    if (b is None):
        b = 0
    return x @ w + b

def sigmoid(z):
    return 1 / (1 + ap.exp(-z))

def relu(z):
    return ap.maximum(0.0, z)


def init_weights(params,seed=None):
    if(seed is not None):
        ap.random.seed(seed)
    d = 0
    for each in params:
        din, dout = each
        d += din * dout

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

        z = fc_forward(a, W, 0)

        if(count < len(params)-1):
            a = sigmoid(z)
            # a = relu(z)
        else:
            if(loss_type is 'l2'):
                # a = sigmoid(z).squeeze()
                # a = relu(z).squeeze()
                a = z.squeeze()
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


######################## Synthetic Data #############################
np.random.seed(0)

n = 2000
d = 500
m = d

A = 3*np.random.rand(n, d)
# A = np.concatenate((A,np.ones((n,1))),axis=1)
b = A@np.random.rand(d) + 0.5*np.random.randn(n)
x0 = 5*np.ones(d)

X_train = A.copy()
y_train = b.copy()
X_val = A[1500:,:]
y_val = b[1500:]

#least squares cost
def ls(x):
    return np.sum(np.power(A@x-b, 2))

#least squares gradient
def lsgrad(x):
    return A.T@(A@x-b)

#least squares sqrt hessian
def lshess_sr(x):
    return A

data = {
        'X_train': X_train,  # training data
        'y_train': y_train,  # training labels
        'X_val': X_val,  # validation data
        'y_val': y_val  # validation labels
    }

x_opt, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

# xstar_gd, opt_gaps_gd, times_gd, numiter_gd, xs_gd = gd(ls, lsgrad, x0)

### Gauss-Newton Solver ###

loss_type = 'l2'

sketch_size = int(0.55*n)
# sketch_size = 1000

input_dim = data['X_train'].shape[1]
hidden_dim = None         ## Use for regression

if(loss_type is 'l2'):
    output_dim = 1
else:
    output_dim = data['y_train'].shape[1]

model,w0 = make_model(input_dim,output_dim,loss_type,hidden_dim,seed=0)

ITER_GN = 10
ITER_GNHS = 50
ITER_GNS = 50
ITER_SGD = 50
n_cg_iter = 50

# lam = 0.0001          ## Use for softmax
lam = 1 / data['X_train'].shape[0] ## use for l2 regression

## Gauss-Newton
w_star_gn, loss_log_gn, val_err_gn, t_solve_gn = gn_test(data, model, lam, w0, loss_type, ITER_GN, n_cg_iter, GN=True,
                                                     iterative = True, backtrack = True)

gaps_gn = np.apply_along_axis(ls, 0, w_star_gn) - ls(x_opt)

## Gauss-Newton Sketch
w_star_gns, loss_log_sketch, val_err_sketch, t_solve_sketch = gn_sketch_test(data, model, lam, w0, loss_type,
                                                                     ITER_GNS, n_cg_iter,
                                                     GN=True, iterative = False,
                                                     backtrack=True, sketch_size=sketch_size)


gaps_gns = np.apply_along_axis(ls, 0, w_star_gns) - ls(x_opt)

# Gauss-Newton Half-Sketch (Iterative)
w_star_gnhs, loss_log_hsketch, val_err_hsketch, t_solve_hsketch = gn_half_sketch_test(data, model, lam, w0, loss_type,
                                                                     ITER_GNHS, n_cg_iter,
                                                     GN=True, iterative = True,
                                                     backtrack=True, sketch_size=sketch_size)
gaps_gnhs = np.apply_along_axis(ls, 0, w_star_gnhs) - ls(x_opt)

# SGD
w_star_sgd, loss_log_sgd, val_err_sgd, t_solve_sgd = gd_test(data, model, lam, w0, loss_type,ITER_SGD,
                                                            GN=False, backtrack=True, sketch_size = sketch_size)


gaps_gd = np.apply_along_axis(ls, 0, w_star_sgd) - ls(x_opt)

import matplotlib.pyplot as plt
plt.plot(np.arange(len(loss_log_gn)) * t_solve_gn / ITER_GN, gaps_gn + 1e-5, 'r^',  np.arange(len(
    loss_log_sketch)) * t_solve_sketch / ITER_GNS, gaps_gns + 1e-5, 'bs', np.arange(len(
    loss_log_hsketch)) * t_solve_hsketch / ITER_GNHS, gaps_gnhs, 'k^',
         np.arange(len(loss_log_sgd)) * t_solve_sgd / ITER_SGD, gaps_gd + 1e-5, 'g--')

plt.yscale('log')
plt.legend(['GN','GN-Sketch', 'GN-Half-Sketch','GD'])
plt.title('Optimality Gap vs Wall Clock Time')
plt.show()


plt.subplot(2,1,1)
plt.semilogy(np.arange(len(loss_log_gn)), loss_log_gn, 'k', np.arange(len(loss_log_sketch)),
             loss_log_sketch, 'g', np.arange(len(loss_log_hsketch)), loss_log_hsketch,
             'r', np.arange(len(loss_log_sgd)), loss_log_sgd, 'b')
plt.title('Training loss')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton', 'Gauss-Newton-Sketch','Gauss-Newton Half-Sketch','SGD'])

plt.subplot(2,1,2)
plt.semilogy(np.arange(len(val_err_gn)), val_err_gn, 'k', np.arange(len(val_err_sketch)), val_err_sketch, 'g',
         np.arange(len(val_err_hsketch)), val_err_hsketch, 'r',
         np.arange(len(val_err_sgd)), val_err_sgd, 'b')
plt.xlabel('Iteration')
plt.title('Validation Loss')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton', 'Gauss-Newton-Sketch','Gauss-Newton Half-Sketch','SGD'])
plt.show()


plt.subplot(2,1,1)
plt.semilogy(np.arange(len(loss_log_gn)) * t_solve_gn / ITER_GN, loss_log_gn, 'k', np.arange(len(
    loss_log_sketch)) * t_solve_sketch / ITER_GNS, loss_log_sketch, 'g', np.arange(len(
    loss_log_hsketch)) * t_solve_hsketch / ITER_GNHS, loss_log_hsketch, 'r', np.arange(len(
    loss_log_sgd)) * t_solve_sgd / ITER_SGD, loss_log_sgd, 'b')
plt.title('Training loss')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton', 'Gauss-Newton-Sketch','Gauss-Newton Half-Sketch','SGD'])


plt.subplot(2,1,2)
plt.semilogy(np.arange(len(val_err_gn)) * t_solve_gn / ITER_GN, val_err_gn, 'k', np.arange(len(
    val_err_sketch)) * t_solve_sketch / ITER_GNS, val_err_sketch, 'g', np.arange(len(
    val_err_hsketch)) * t_solve_hsketch / ITER_GNHS, val_err_hsketch, 'r',  np.arange(len(
    val_err_sgd)) * t_solve_sgd / ITER_SGD, val_err_sgd, 'b')
plt.xlabel('Seconds')
plt.title('Validation loss')
plt.grid(True, which='both')
plt.legend(['Gauss-Newton', 'Gauss-Newton-Sketch','Gauss-Newton Half-Sketch','SGD'])
plt.show()



print("I'm done")
