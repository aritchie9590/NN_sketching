import autograd.numpy as ap
from autograd import make_vjp
from autograd import make_jvp
from autograd import grad
from autograd import elementwise_grad
import time
import numpy as np

#Synthetic data
ap.random.seed(0)
n = 50000
d = 784
X = ap.random.randn(n,d)
y = ap.random.randn(n)

#Define single-layer sigmoid-activated network
def f(w,X):
    z = X @ w
    a = 1 / (1 + ap.exp(-z))
    return a

w = ap.random.randn(d) #random weights

print('Testing JVPs ...')
#Analytical Soln

#Verify ROP with analytical solution
def sigmoid(u):
    return 1 / (1 + np.exp(-u))

start = time.time()
q = X @ w
c = (sigmoid(q)*(1-sigmoid(q)))
J = X * np.expand_dims(c,axis=1)

v = ap.random.randn(d)  #Some dummy vector

jvp_out2 = J @ v
end = time.time()
t_analytical = end - start
print('Time to compute analytical JVP: {:.3f} s'.format(t_analytical))


# Define ROP (JVP)
jvp = make_jvp(f)(w,X)   #Jacobian-vector product (JVP) function for the network

time_jvp = 0
for i in range(0,100):
    v = ap.random.randn(d)  # Some dummy vector
    start = time.time()
    jvp_out = jvp(v)[1]           #Compute jvp of the dummy vector
    end = time.time()
    time_jvp += end - start

time_jvp /= 100
print('Avg time to compute fast JVP 100 trials: {:.3f} s'.format(time_jvp))
print('Avg speedup of fast JVP to analytical: {:.3f} \n'.format(t_analytical / time_jvp))
print('Testing sketched JVPs ...')
# assert np.linalg.norm(jvp_out - jvp_out2) < 1e-10

#Sketched Analytical
m = int(n/2)
Xs = X[:m,:]
start = time.time()
q = Xs @ w
c = (sigmoid(q)*(1-sigmoid(q)))
J = Xs * np.expand_dims(c,axis=1)

v = ap.random.randn(d)  #Some dummy vector

jvp_out2 = J @ v
end = time.time()
t_s_analytical = end - start

# Sketched fast JVP
jvp_sketch = make_jvp(f)(w,X[:m,:])

time_s_jvp = 0
for i in range(0,100):
    v = ap.random.randn(d)  # Some dummy vector
    start = time.time()
    s_jvp_out = jvp_sketch(v)[1]
    end = time.time()
    time_s_jvp += end - start

time_s_jvp /= 100

print('Time to compute sketched analytical JVP: {:.3f} s'.format(t_s_analytical))
print('Speedup of sketched analytical JVP: {:.3f} '.format(t_analytical / t_s_analytical))

print('Avg time to compute sketched fast JVP 100 trials: {:.6f} s'.format(time_s_jvp))
print('Avg speedup of sketched JVP: {:.3f} '.format(time_jvp / time_s_jvp))
print('Avg speedup of sketched JVP to sketched analytical: {:.3f} \n'.format(t_s_analytical / time_s_jvp))

# s_jvp_out2 = S @ J @ v


# assert np.linalg.norm(s_jvp_out - s_jvp_out2) < 1e-10


# def objective(w,inputs,y):
#     return 0.5*ap.sum((f(w,inputs) - y)**2)

# Compute gradients of the network wrt weights
# grad_f = grad(objective)    #gradient function

#########################################################################
# #Define LOP (VJP)
# def lop_fxn(w,inputs,vector):
#     return f(w,inputs) * vector
#
# lop = elementwise_grad(lop_fxn)
#
# v2 = np.random.randn(n)
# start = time.time()
# lop_out = lop(w,X,v2)
# end = time.time()
# print('Time to compute VJP (grad): {:.3f} s'.format(end-start))
#
# lop_out2 = J.T @ v2
#
# assert np.linalg.norm(lop_out - lop_out2) < 1e-10

######################## Analytical VJP ##################################
print('Testing VJPs...')

v = np.random.randn(n)

start = time.time()
q = X @ w
c = (sigmoid(q)*(1-sigmoid(q)))
J = X * np.expand_dims(c,axis=1)
vjp_analytical = J.T @ v
end = time.time()
t_vjp_analytical = end - start

print('Time to compute analytical VJP: {:.3f} s'.format(t_vjp_analytical))

######################## Autograd make_vjp ################################
vjp = make_vjp(f)(w,X)[0]
vjp_out = vjp(v)

assert np.linalg.norm(vjp_out - vjp_analytical) < 1e-10


time_vjp = 0
for i in range(0,100):
    v = np.random.randn(n)
    start = time.time()
    vjp_out = vjp(v)
    end = time.time()
    time_vjp += end - start

time_vjp /= 100

print('Avg time to compute fast VJP 100 trials: {:.3f} s'.format(time_vjp))
print('Avg speedup of fast VJP to analytical: {:.3f} \n'.format(t_vjp_analytical / time_vjp))


# s_lop_out = (S @ J).T @ v3

print('Testing sketched VJPs ...')

# Sketched analytical VJP
v = np.random.randn(m)

start = time.time()
q = Xs @ w
c = (sigmoid(q)*(1-sigmoid(q)))
J = Xs * np.expand_dims(c,axis=1)
vjp_analytical = J.T @ v
end = time.time()
t_vjp_analytical = end - start

print('Time to compute sketched analytical VJP: {:.3f} s'.format(t_vjp_analytical))

# Sketched fast VJP

s_vjp = make_vjp(f)(w,X[:m,:])[0]

time_s_vjp = 0
for i in range(0,100):
    v = np.random.randn(m)
    start = time.time()
    s_vjp_out = s_vjp(v)
    end = time.time()
    time_s_vjp += end - start

time_s_vjp /= 100

print('Avg time to compute Sketched VJP: {:.6f} s'.format(time_s_vjp))
print('Avg speedup of sketched VJP: {:.3f} '.format(time_vjp / time_s_vjp))
print('Avg speedup of sketched fast VJP to sketched analytical {:.3f}'.format(t_vjp_analytical / time_s_vjp))


# assert np.linalg.norm(s_lop_out - s_vjp_out) < 1e-10


# #Compute gradients wrt weights using LOP
# grads = lop(w,X,f(w,X) - y)
# grads2 = J.T @ (f(w,X) - y)
#
# assert np.linalg.norm(grads - grads) < 1e-10


## Compute S @ J @ v where S is an unstructured matrix

# start = time.time()
# S = np.random.randn(m,n)
# end = time.time()
# print('Time to form S: {:.3f} s'.format(end-start))
#
# v = np.random.randn(d)
# jvp = make_jvp(f)(w,X)
#
# start = time.time()
# jvp_out = jvp(v)[1]
# end = time.time()
# print('Time to compute JVP {:.3f} s'.format(end-start))
#
# start = time.time()
# y = S @ jvp_out
# end = time.time()
#
# print('Time to compute S*JVP(v): {:.3f} s'.format(end-start))

from autograd import make_ggnvp

ggnvp = make_ggnvp(f)(w,X)

v =  np.random.randn(d)
ggnvp_out = ggnvp(v)

vjp = make_vjp(f)(w,X)[0]
jvp = make_jvp(f)(w,X)
ggnvp_out2 = vjp(jvp(v)[1])


print("I'm done")