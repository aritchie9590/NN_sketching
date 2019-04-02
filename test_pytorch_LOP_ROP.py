import numpy

import torch
import torch.nn as nn 

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

#Computing a vector-Jacobian product (LOP)
v = torch.randn((batch_size,output_dim))
v.requires_grad = True 

vJ, = torch.autograd.grad(res, x, v, create_graph=True)
print('v.T * J: {}  \n{}'.format(vJ, vJ.shape))

#Computing a Jacobian-vector product (ROP) - using vjp trick
u = torch.ones((batch_size,input_dim))
Ju, = torch.autograd.grad(vJ, v, u)
print('J * u: {} \n{}'.format(Ju, Ju.shape))

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
