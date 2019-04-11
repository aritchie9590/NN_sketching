import numpy as np
from scipy.optimize import minimize
import time
from autograd import make_vjp
from autograd import make_jvp
from autograd import make_ggnvp
import matplotlib.pyplot as plt
import autograd.numpy as ap

from autograd import grad
from autograd import elementwise_grad

def solver(data, f, lam, w0, loss_type, ITERNEWTON, n_cgiter=None, backtrack=True, sketch_size=None):

    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']

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

    print(np.linalg.norm(f(w, X_train) - y_train) ** 2 / n + 0.5*lam*np.linalg.norm(w)**2)
    start = time.time()
    for iter in range(0, ITERNEWTON):


        sample_idx = np.random.choice(n,sketch_size,False)
        Xs = X_train[sample_idx, :]

        vjp = make_vjp(f)(w, X_train)[0]
        e = f(w, X_train) - y_train
        loss = 0.5 * np.linalg.norm(e) ** 2 / n    # compute loss

        g = vjp(e) / n + lam * w

        jvp = make_jvp(f)(w, Xs)

        def ggnvp_fxn(v):
            pad = np.zeros(n)
            pad[sample_idx] = jvp(v)[1]
            return vjp(pad) / n + lam * v

        subprob = lambda v: g.T @ v + 0.5 * v.T @ ggnvp_fxn(v)
        subgrad = lambda v: g + ggnvp_fxn(v)

        res = minimize(subprob, w, jac=subgrad, method="CG", \
                       options={'gtol': 1e-3, 'norm': 2.0, 'eps': 0.1, \
                                'maxiter': n_cgiter, 'disp': False})
        dw = res.x

        decr = np.sum(dw * g)
        if(backtrack):

            # Perform backtracking line-search
            val = loss + 0.5*lam*np.linalg.norm(w)**2

            t = 1
            alpha = backtrack_alpha
            beta = backtrack_beta

            bts = 0
            while (0.5*np.linalg.norm(f(w + t * dw,X_train) - y_train)**2/n + 0.5*lam*np.linalg.norm(w + t*dw)**2 >
                   val + alpha * t * decr):
                t = beta * t
                bts +=1
                if(bts > 50):
                    print("Reached maximum backtracking")
                    break
        else:
            t = 1 / (iter + 1)

        # Update the NN params
        w = w + t * dw
        w_log[:,iter + 1] = w

        val_err[iter +1] = np.linalg.norm(f(w,X_val) - y_val)**2 / n

        loss = np.linalg.norm(f(w, X_train) - y_train) ** 2 / n + 0.5 * lam * np.linalg.norm(w)**2
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