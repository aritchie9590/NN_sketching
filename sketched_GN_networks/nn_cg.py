import numpy as np

def nn_cgm(ggnvp,b,x0,niters,record=False):

    cost = lambda v:  0.5*v.T @ (ggnvp(v)) - b.T@v

    cost_log = np.zeros(niters)

    x = x0.copy()
    r = b - ggnvp(x)
    p = r

    rsold = r.T @ r

    for i in range(0,niters):
        Ap = ggnvp(p)
        alpha = rsold / (p.T @ Ap)

        x = x + alpha * p

        if(record):
            cost_log[i] = cost(x)
        else:
            cost_log[i] = 0

        r = r - alpha * Ap
        rsnew = r.T @ r
        if np.sqrt(rsnew) < 1e-10:
            break

        beta = rsnew / rsold

        p = r + beta * p

        rsold = rsnew

    return x, cost_log