import numpy

import torch
import torch.nn as nn 

A = torch.randn((3,3))
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

'''
#Using PyTorch NN Layers
batch_size = 1 
input_dim = 12
sketch_dim = int(input_dim/2)
output_dim = 1

x = torch.randn((batch_size, input_dim), requires_grad=True)
y = torch.ones((batch_size, output_dim))

w = nn.Linear(input_dim, output_dim) #weight matrix
sig = nn.Sigmoid() #linear activation

#forward pass
y_hat = sig(w(x))
res = y_hat - y #residual
'''

'''
#Now with sketching matrix
S = torch.randn((sketch_dim, batch_size))
M = torch.zeros((sketch_dim, input_dim))

#forward pass
y_hat = sig(w(x))

last = torch.zeros(v.shape)
for m in range(sketch_dim):
    y_hat.backward(S[m,None].transpose(0,1), retain_graph=True)

    M[m] = next(w.parameters()).grad - last
    last = M[m]

print('{} \n Sketched Jacobian: {}'.format(M.shape, M))
'''
