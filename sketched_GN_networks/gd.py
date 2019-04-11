import numpy as np
import time
from autograd import make_vjp
from autograd import make_jvp
from autograd import make_ggnvp
import matplotlib.pyplot as plt
import autograd.numpy as ap

from autograd import grad
from autograd import elementwise_grad

def solver(data, f, lam, w0, loss_type, ITERNEWTON, stochastic = True, backtrack=True, sketch_size=None):

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

    max_backtrack = 50
    backtrack_alpha = 0.2
    backtrack_beta = 0.5
    NTOL = 1e-8

    w = w0.copy()

    w_log = np.zeros((len(w0),ITERNEWTON + 1))
    w_log[:,0] = w
    loss_log = np.zeros(ITERNEWTON + 1)
    val_err = np.zeros(ITERNEWTON + 1)

    n = X_train.shape[0]

    loss_log[0] = 0.5*np.linalg.norm(f(w, X_train) - y_train)**2 / n + lam * np.linalg.norm(w) ** 2 / 2
    val_err[0] = 0.5*np.linalg.norm(f(w, X_val) - y_val)**2 / n

    randperm = lambda n: np.random.permutation(n)
    randp = np.hstack((randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), \
                       randperm(n),
                       randperm(n), randperm(n), randperm(n), randperm(n)))

    print(np.linalg.norm(f(w, X_train) - y_train) ** 2 / n + 0.5*lam*np.linalg.norm(w)**2)
    start = time.time()
    for iter in range(0, ITERNEWTON):

        if(stochastic):

            sample_idx = randp[sketch_size * (iter) + 1: sketch_size * (iter + 1) + 1]
            Xs = X_train[sample_idx, :]
            vjp = make_vjp(f)(w, Xs)[0]
            e = f(w, Xs) - y_train[sample_idx]
            loss = np.linalg.norm(e) ** 2 / n  # compute loss
            grads = vjp(e) / n

        else:

            vjp = make_vjp(f)(w, X_train)[0]
            e = f(w, X_train) - y_train
            loss = np.linalg.norm(e) ** 2   # compute loss
            grads = vjp(e) + lam * w # compute gradient

        dw = - grads

        # Perform backtracking line-search
        val = loss + 0.5*lam*np.linalg.norm(w)**2

        if(backtrack):
            fprime = grads.T @ dw

            t = 1

            alpha = backtrack_alpha
            beta = backtrack_beta

            bts = 0

            if(stochastic):

                while (np.linalg.norm(f(w + t * dw, Xs) - y_train[sample_idx]) ** 2 / n > val + alpha * t * fprime):
                    t = beta * t
                    bts +=1
                    if(bts > 50):
                        print("Reached maximum backtracking iterations")
                        break

            else:
                while (np.linalg.norm(f(w + t * dw,X_train) - y_train)**2 + 0.5*lam*np.linalg.norm(w + t * dw)**2 > val +
                       alpha * t *
                       fprime):
                    t = beta * t
                    bts += 1
                    if (bts > 50):
                        print("Reached maximum backtracking iterations")
                        break
        else:
            t = 3000 / (iter + 1)
            # t = 0.01

        ## Update the NN params

        w = w + t * dw
        w_log[:,iter + 1] = w

        val_err[iter +1] = 0.5*np.linalg.norm(f(w,X_val) - y_val)**2/n

        loss = np.linalg.norm(f(w, X_train) - y_train) ** 2 / n + 0.5*lam*np.linalg.norm(w)**2
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