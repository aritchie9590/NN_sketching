import numpy

import torch
import torch.nn as nn 

batch_size = 10
input_dim = 12
sketch_dim = int(input_dim/2)
output_dim = 1

#Backward pass with residual
x = torch.randn((batch_size, input_dim), requires_grad=True)
y = torch.ones((batch_size, output_dim))

w = nn.Linear(input_dim, output_dim) #weight matrix
sig = nn.Sigmoid() #linear activation

#forward pass
y_hat = sig(w(x))
res = y_hat - y #residual

y_hat.backward(res)
v = next(w.parameters()).grad 

#Backward pass with sketching matrix
S = torch.randn((sketch_dim, batch_size))
M = torch.zeros((sketch_dim, input_dim))

#forward pass
y_hat = sig(w(x))

last = torch.zeros(v.shape)
for m in range(sketch_dim):
    y_hat.backward(S[m,None].transpose(0,1), retain_graph=True)

    M[m] = next(w.parameters()).grad - last
    last = M[m]

print('residual * Jacobian {}  \n: {}'.format(v.shape, v))
print('{} \n Sketched Jacobian: {}'.format(M.shape, M))
