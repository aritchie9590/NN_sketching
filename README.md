# Instructions - How to run code:

### Datasets: 
- MNIST : Download directly through PyTorch
- CIFAR : Download directly through PyTorch
- CT Dataset: https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis

### Files to run:
- `ct_test.py` : Use default settings to run a multi-layered feed-forward network on the CT dataset (regression). The optimization is performed using GN, GN Half-sketch, GN Sketch, SGD, and ADAM. 
Optional to run on GPU with 'cuda' parameter
- `mnist_cnn_test.py` : Use default settings to run a convolution + 2 fc layer network on MNIST for 10 digit classification. The optimization is performed using GN, GN Half-sketch, GN Sketch, SGD, and ADAM.
Recommended to run on CPU only, 12 GB GPU runs out of memory
- `sketched_GN_networks.py` : Runs the mnist binary regression experiment using the python implementation in autograd (not PyTorch). Use default settings to run a 2 fc layer network on MNIST for 2 digit (0, 1) classification. 

### Other files:
- `GN_solver.py` contains the python implementation of the Gauss-Newton Sketch solver

# NN_sketching

### Papers

#### Sketching
1. [Iterative Hessian Sketch: Fast and Accurate SolutionApproximation for Constrained Least-Squares](http://www.jmlr.org/papers/volume17/14-460/14-460.pdf) M. Pilanci, M. J. Wainwright, JMLR 2016
2. [Information-TheoreticMethods in Data Science: Information-theoretic bounds on sketching](http://web.stanford.edu/~pilanci/papers/infosketch.pdf) M. Pilanci, Stanford

#### Second-Order optimization methods
1. [Training Feedforward Networks with the  Marquardt Algorithm ](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=329697) M. T. Hagan, M. B. Menhaj, IEEE Transactions on Neural Networks 1994
2. [Fast Curvature Matrix-Vector Products for Second-Order Gradient Descent](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=0258AD1C6C96F0DC0BCB0F456F5BCF88?doi=10.1.1.421.1443&rep=rep1&type=pdf) N. N. Schraudolph, Neural Computation 2002
3. [Practical Gauss-Newton Optimisation for Deep Learning](https://arxiv.org/pdf/1706.03662.pdf) A. Botev, H. Ritter, D. Barber, ICML 2017
4. [Deep learning via Hessian-free optimization](http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf) J. Martens, ICML 2010
5. [First-and second-order methods for learning: between steepest descent and Newton's method.](https://www-mitpressjournals-org.proxy.lib.umich.edu/doi/pdf/10.1162/neco.1992.4.2.141) R. Battiti, Neural Computation 1992

### Weblinks

#### Jacobian-Vector products
1. [A new trick for calculating Jacobian vector products](https://j-towns.github.io/2017/06/12/A-new-trick.html)
2. [Automatic Differentiation](https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf)
3. [PyTorch Autograd](https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95)
4. [Computing the Jacobian matrix of a neural network in Python](https://medium.com/unit8-machine-learning-publication/computing-the-jacobian-matrix-of-a-neural-network-in-python-4f162e5db180)
