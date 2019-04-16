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

    # loss_log[0] = 0.5*np.linalg.norm(f(w, X_train) - y_train)**2 / n + lam * np.linalg.norm(w) ** 2 / 2
    # val_err[0] = 0.5*np.linalg.norm(f(w, X_val) - y_val)**2 / n

    if (loss_type is 'l2'):
        loss_log[0] = 0.5 * np.linalg.norm(f(w, X_train) - y_train) ** 2 / n + lam * np.linalg.norm(w) ** 2 / 2
        val_err[0] = 0.5 * np.linalg.norm(f(w, X_val) - y_val) ** 2 / n + lam * np.linalg.norm(w) ** 2 / 2


    else:  # softmax
        loss_log[0] = 0.5 * np.sum(np.linalg.norm(softmax(f(w, X_train)) - y_train,axis=1)**2) / n + lam * \
                      np.linalg.norm(
            w) ** 2 / 2
        val_err[0] = np.sum(abs(np.argmax(softmax(f(w, X_val)), axis=1) - np.argmax(y_val, axis=1)) > 0,
                            axis=0) / y_val.shape[0]

    randperm = lambda n: np.random.permutation(n)
    randp = np.hstack((randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), randperm(n), \
                       randperm(n),
                       randperm(n), randperm(n), randperm(n), randperm(n)))

    print("Loss function: {:.5f}, Validation error: {:.5f}".format(loss_log[0], val_err[0]))
    start = time.time()
    for iter in range(0, ITERNEWTON):

        if(stochastic):

            sample_idx = randp[sketch_size * (iter) + 1: sketch_size * (iter + 1) + 1]
            Xs = X_train[sample_idx, :]
            vjp = make_vjp(f)(w, Xs)[0]

            if (loss_type is 'l2'):  # Regression
                e = f(w, Xs) - y_train[sample_idx]

            else:  # Softmax
                e = softmax(f(w, Xs)) - y_train[sample_idx]

        else:

            vjp = make_vjp(f)(w, X_train)[0]
            if (loss_type is 'l2'):  # Regression
                e = f(w, X_train) - y_train

            else:  # Softmax
                e = softmax(f(w, X_train)) - y_train


        grads = vjp(e) / n + lam * w # compute gradient

        dw = - grads

        loss = np.linalg.norm(e) ** 2 / n  # compute loss

        # Perform backtracking line-search
        val = loss + 0.5*lam*np.linalg.norm(w)**2

        fprime = grads.T @ dw

        if(backtrack):

            if(stochastic):

                if (loss_type is 'l2'):
                    fxn_val = lambda v: np.linalg.norm(f(v, Xs) - y_train[sample_idx]) ** 2 / n + lam * \
                                        np.linalg.norm(
                        v) ** 2 / 2
                else:
                    fxn_val = lambda v: np.linalg.norm(
                        softmax(f(v, Xs)) - y_train[sample_idx])**2 / n + lam * np.linalg.norm(v) ** 2 / 2


                t = 1

                alpha = backtrack_alpha
                beta = backtrack_beta

                bts = 0

                while (fxn_val(w + t * dw) > val + alpha * t * fprime):
                    t = beta * t
                    bts +=1
                    if(bts > 50):
                        print("Reached maximum backtracking iterations")
                        break

            else:
                if (loss_type is 'l2'):
                    fxn_val = lambda v: np.linalg.norm(f(v, X_train) - y_train) ** 2 / n + lam * np.linalg.norm(
                        v) ** 2 / 2
                else:
                    fxn_val = lambda v: np.linalg.norm(
                        softmax(f(v, X_train)) - y_train)**2 / n + lam * np.linalg.norm(v) ** 2 / 2

                t = 1

                alpha = backtrack_alpha
                beta = backtrack_beta

                bts = 0

                while (fxn_val(w + t * dw) > val + alpha * t * fprime):
                    t = beta * t
                    bts += 1
                    if (bts > 50):
                        print("Reached maximum backtracking iterations")
                        break
        else:
            # t = 10 / (iter + 1)
            t = 100 / (iter + 1)

        ## Update the NN params

        w = w + t * dw
        w_log[:,iter + 1] = w


        if (loss_type is 'l2'):
            loss = np.linalg.norm(f(w, X_train) - y_train) ** 2 / n + 0.5 * lam * np.linalg.norm(w) ** 2
            loss_log[iter + 1] = loss

            val_err[iter + 1] = 0.5 * np.linalg.norm(f(w, X_val) - y_val) ** 2 / n + lam * np.linalg.norm(w) ** 2 / 2

        else:
            loss = np.linalg.norm(softmax(f(w, X_train)) - y_train)**2 / n + 0.5 * lam * \
                   np.linalg.norm(w) ** 2
            loss_log[iter + 1] = loss

            val_err[iter + 1] = np.sum(
                abs(np.argmax(softmax(f(w, X_val)), axis=1) - np.argmax(y_val, axis=1)) > 0,
                axis=0) / y_val.shape[0]

        if np.sum(np.power(dw, 2)) <= NTOL :
            loss_log = loss_log[:iter]
            val_err = val_err[:iter]
            w_log = w_log[:,:iter]
            break

        print("Loss function: {:.5f}, Validation error: {:.5f}".format(loss,val_err[iter+1]))

    end = time.time()
    t_solve = end - start

    return w_log, loss_log, val_err, t_solve

def softmax(x):
  # exps = ap.exp(x)
  exps = np.exp(x - np.max(x))
  sum_exps = ap.sum(exps,axis=1,keepdims=True)
  S = exps / sum_exps
  return S