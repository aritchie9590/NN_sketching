import numpy

import torch
import torch.nn as nn 
import torch.nn.functional as F

A = torch.randn((7,3))
x = torch.randn((3,1))
x.requires_grad = True

#Computing a vector-Jacobian product (LOP)
f = torch.mm(A,x)

v = torch.randn(f.shape)
vJ, = torch.autograd.grad(f, x, v)
print('Expected: {}'.format(v.transpose(0,1)@A))
print('Computed: v.T * J: {}'.format(vJ))

#Computing a Jacobian-vector product (ROP) - using vjp trick
f = torch.mm(A,x)

u = torch.randn(f.shape)
u.requires_grad = True 
uJ, = torch.autograd.grad(f, x, u, create_graph=True)

v = torch.ones(x.shape)
Jv, = torch.autograd.grad(uJ, u, v)
print('Expected: {}'.format(A@v))
print('J * v: {}'.format(Jv))

#Computing the Gauss-Newton-vector product 
f = torch.mm(A,x)

u = torch.ones(f.shape)
u.requires_grad = True 

uJ, = torch.autograd.grad(f, x, u, create_graph=True)

v = torch.randn(x.shape)
v.requires_grad = True
Jv, = torch.autograd.grad(uJ, u, v, create_graph=True)

JtJv, = torch.autograd.grad(Jv, v, Jv)
print('Expected: {}'.format(A.transpose(0,1)@A@v))
print('JtJv: {}'.format(JtJv))

#Using NN Layers
n = 1 #mini-batch size
d1 = 7 #input dimension
d2 = 3 #hidden dimension
o = 1 #output dimension
sketch_dim = int(d1/2)

torch.manual_seed(0)
X = torch.randn((n,d1))
W1 = torch.randn((d2,d1))
b1 = torch.ones(d2)
W2 = torch.randn((o,d2))
b2 = torch.ones(o)

W1.requires_grad = True
b1.requires_grad = True
W2.requires_grad = True
b2.requires_grad = True

a = torch.sigmoid(F.linear(X,W1,b1))
f = F.linear(a,W2,b2)

u = torch.ones(f.shape)
u.requires_grad = True 

import pdb; pdb.set_trace()
grad_params = torch.autograd.grad(f, [W1,b1,W2,b2], u, create_graph=True)

u_W1 = torch.ones(W1.shape)
u_b1 = torch.ones(b1.shape)
u_W2 = torch.ones(W2.shape)
u_b2 = torch.ones(b2.shape)
u_W1.requires_grad = True 
u_b1.requires_grad = True 
u_W2.requires_grad = True 
u_b2.requires_grad = True 

Jv_params = torch.autograd.grad(grad_params, [u,u,u,u], [u_W1,u_b1,u_W2,u_b2], create_graph=True)

for uJ in grad_params:
    print('Param shape: {}'.format(uJ.shape))
    v = torch.ones(uJ.shape)
    v.requires_grad = True
    Jv, = torch.autograd.grad(uJ, u, v, create_graph=True)

    JtJv, = torch.autograd.grad(Jv, v, Jv)
