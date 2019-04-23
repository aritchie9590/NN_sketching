#PyTorch optimizer that uses a 2nd order method (Gauss-Newton method) to solve for optimal parameters in a network

import torch
from functools import reduce
import torch.nn as nn
from torch.optim.optimizer import Optimizer

"""
Compute Gauss-Newton vector product
Args:
    f: output from cost function
    w: take derivative w.r.t to parameters of network
    v: vector multiply with Gauss-Newton approx
"""
def _make_ggnvp(err, params, w0, n, reg, idx):
    u = err.clone()

    grad_params = torch.autograd.grad(err, params, u, create_graph=True) #gradients
    grad = torch.nn.utils.parameters_to_vector(grad_params)
    grad = grad/n + reg*w0 #average gradients + regularize

    grad_sketch_params = torch.autograd.grad(err[idx], params, u[idx], create_graph=True) #sketched gradients; if no sketching, then this is equivalent to grad_params
    
    def ggnvp(w):
        assert w.requires_grad, 'variable must have requires_grad as True'
        JtJv = []
        offset = 0
        for uJ in grad_sketch_params:
            #Separate weights into corresponding elements and then reshape into shape uJ
            numel = uJ.numel()
            v = w[offset:offset+numel].view_as(uJ)
            offset += numel

            Jv, = torch.autograd.grad(uJ, u, v, create_graph=True) #Jacobian vector-product
            temp, = torch.autograd.grad(Jv, v, Jv, create_graph=True) #Jacobian.T @ Jacobian vector-product

            #Accumulate JtJv for all parameters into this list
            JtJv.append(temp)

        JtJv = torch.nn.utils.parameters_to_vector(JtJv)
        JtJv = JtJv/n + reg*w #average JtJv + regularize

        return JtJv

    return grad, ggnvp 

#solve for optimal step direction using the Conjugate Gradient Descent method
def _conjugate_gradient(ggnvp, grad, max_iters):
    n_iter = 0

    #cost = lambda v: 0.5*v @ (ggnvp(v)) - grad@v

    cost_log = torch.zeros(max_iters)
    try:
        device = grad.get_device()
    except:
        device = 'cpu'
    w0 = torch.zeros(grad.shape).to(device)
    w0.requires_grad = True 

    w = w0.clone() #not necessary, but in-place operations can't be done on leaf variables
    r = grad - ggnvp(w)
    p = r

    rs_old = r @ r #r.T @ r
    while n_iter < max_iters:
        Ap = ggnvp(p)
        alpha = rs_old / (Ap @ p) #find optimal step size

        w.add_(alpha.item(), p) #update parameters

        r.sub_(alpha.item(), Ap)
        rs_new = r @ r

        #cost_log[n_iter] = cost(w)

        if torch.sqrt(rs_new) < 1e-10:
            break

        #update direction
        beta = rs_new/rs_old
        p = r + beta * p

        rs_old = rs_new
        n_iter += 1

    return w, cost_log

class GN_Solver(Optimizer):

    """
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): initial guess for learning rate (step size)
        max_iter (int, optional): maximum number of iterations before terminating update on optimization step
        reg (float, optional): regularizer/penalty on weights 
        backtrack (int, optional):
        backtrack_param (tuple, optional): (backtrack_alpha, backtrack_beta)
        tolerance (float, optional):
        sketch_size (int, optional): size of randomized sketching sampling matrix i.e. Number of elements to sample

    """
    def __init__(self, params, lr=1, max_iter=20, reg=0.0, backtrack=50, backtrack_param = (0.2, 0.5), tolerance=1e-8, sketch_size=None):

        if backtrack < 0:
            raise ValueError('Invalid backtrack amount: {}'.format(backtrack))

        defaults = dict(lr=lr, max_iter=max_iter, reg=reg, backtrack=backtrack, bt_alpha=backtrack_param[0], bt_beta=backtrack_param[1], sketch_size=sketch_size)

        super(GN_Solver, self).__init__(params, defaults)

        #TODO: Get details on what parameter groups are
        if len(self.param_groups) != 1:
            raise ValueError("Does not support per-parameter options (parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self.grad_update = 0 #Keep track of the number of updates, used for decreasing step size

    #TODO: This may not be needed
    def __setstate__(self, state):
        super(GN_Solver, self).__setstate__(state)

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)

        return self._numel_cache

    #Reshapes all model parameters to be updated together
    #after gathering the gradients for all parameters
    def _gather_flat_grad(self, reg):
        views_data = [] #weights
        views = [] #gradients
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                d_p = p.grad.to_dense()
                d_p.add(reg, p)
                view = d_p.view(-1)
            else:
                d_p = p.grad 
                d_p.add_(reg, p)
                view = d_p.view(-1)
            
            views_data.append(p.view(-1))
            views.append(view)

        return torch.cat(views_data,0), torch.cat(views, 0)
    
    #update the model weights
    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel

        assert offset == self._numel()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss
        """
        orig_loss,err,pred = closure()
        loss = orig_loss

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        reg = group['reg']
        backtrack = group['backtrack']
        bt_alpha = group['bt_alpha']
        bt_beta = group['bt_beta']
        sketch_size = group['sketch_size']

        #import pdb; pdb.set_trace()
        n = err.shape[0] #batch size
        #If sketching the jacobian, randomly select [sketch_size] samples
        
        if sketch_size is not None:
            idx = torch.randperm(n)[:sketch_size]
        else:
            idx = torch.arange(n) #Don't sketch, use all samples
       
        w0 = nn.utils.parameters_to_vector(self._params) #weight parameters in vector form
        
        #Compute Gauss-Newton vector product 
        grad, ggnvp = _make_ggnvp(err,self._params,w0,n,reg,idx) #return gradient in vector form + ggnvp function
        #Solve for the Conjugate Gradient Direction
        dw, cost_log = _conjugate_gradient(ggnvp, grad,max_iter)

        #Perform backtracking line search
        val = loss + 0.5 * reg * torch.norm(w0)**2
        fprime = -1*dw @ grad
        
        self.grad_update += 1
        if backtrack > 0:
            t = lr

            #TODO: If using backtracking, get new loss with (w0 - t*dw) as network parameters
            bts = 0
            alpha = bt_alpha
            beta = bt_beta 
            while (loss + 0.5 * reg * torch.norm(w0 - t*dw)**2 > val + alpha * t * fprime):
                t = beta * t
                bts += 1
                if bts > backtrack:
                    print('Maximum backtracking reached, accuracy not guaranteed')
                    break
        else: #use a decreasing step-size
            t = 1/self.grad_update

        #Update the model parameters
        self._add_grad(-t, dw)
        
        return val, pred
