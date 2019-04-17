import os 
import argparse
import numpy as np 
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from GN_Solver import GN_Solver

parser = argparse.ArgumentParser()

#settings
parser.add_argument('--dataset', default='mnist', type=str, help='dataset(s) to use')
default_datasets = ['mnist', 'cifar10']
parser.add_argument('--two_layer', action='store_true', help='add parameter to use two-layer architecture instead of single-layer')
parser.add_argument('--max_iter', default=20, type=int, help='max num of iterations to train on')
parser.add_argument('--batch_size', default=60000, type=int, help='mini-batch size') 

parser.add_argument('--sketch_size', default=1500, type=int, help='sketch size for sketching algorithms')
#parser.add_argument('--reg', default=0.0001, type=float, help='regularizer')

parser.add_argument('--cuda', action='store_true', help='use gpu')

parser.set_defaults(cuda=False)
#parser.set_defaults(two_layer=False)
parser.set_defaults(two_layer=True)
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

#simple feed-forward neural network
class Model(nn.Module):
    
    def __init__(self, input_size=784, hidden_dim=200, output_dim=1, two_layer=False):
        super().__init__()

        self.single_layer = nn.Sequential(
                            nn.Linear(input_size, 1, bias=True),
                            nn.Sigmoid())

        self.two_layer = nn.Sequential(
                         nn.Linear(input_size, hidden_dim),
                         nn.Sigmoid(), #Or ReLU()
                         nn.Linear(hidden_dim, 1),
                         nn.Sigmoid())

        #Set-up feed-forward network as two layer or single layer
        if two_layer:
            self.ff_network = self.two_layer
            self.single_layer = None
        else:
            self.ff_network = self.single_layer
            self.two_layer = None

    def forward(self, x):
        out = self.ff_network(x)
        #out = self.single_layer(x)

        return out


class VGG(nn.Module):
    # You will implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # Since the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3, 
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            # TODO: convolutional layer, input channels 3, output channels 8, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(1, 2, kernel_size=3, padding=1),#, stride=1,padding=1,bias=True),
            nn.MaxPool2d(2),
            
            
        
            # Stage 2
            # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
            # TODO: max-pooling layer, size 2
            
            nn.Conv2d(2, 4, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            # Stage 3
            # TODO: convolutional layer, input channels 16, output channels 32, filter size 3
            # TODO: convolutional layer, input channels 32, output channels 32, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            #nn.Conv2d(32, 32, kernel_size=3, padding=1),
            #nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            
            # Stage 3
            # TODO: convolutional layer, input channels 16, output channels 32, filter size 3
            # TODO: convolutional layer, input channels 32, output channels 32, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            #nn.Conv2d(32, 32, kernel_size=3, padding=1),
            #nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
            )
        
        self.fc = nn.Sequential(
            # TODO: fully-connected layer (64->64)
            nn.Sigmoid(),
            nn.Linear(16,16),
            nn.Sigmoid(),
            # TODO: fully-connected layer (64->10)
            nn.Linear(16, 1),
            nn.Sigmoid()
            #nn.Softmax()
        )


    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        x = x.view(-1, 16)
        x = self.fc(x)
        return x

#We will only perform binary classification between two labels
def filter_class_labels(dataset):
    #Filter out all digits except these two
    l1 = 0
    l2 = 1
   
    if dataset.train:
        idx_l1 = dataset.targets == l1
        idx_l2 = dataset.targets == l2 
        
        idx = idx_l1 + idx_l2
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
    else:
        idx_l1 = dataset.test_labels == l1
        idx_l2 = dataset.test_labels == l2

        idx = idx_l1 + idx_l2
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

def get_datasets(args):
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data/', train=True,
        download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='./data/', train=False,
        download=True, transform=transforms.ToTensor())
    elif args.dataset == 'cifar10':
        
        train_dataset = datasets.CIFAR10(root='./data',
        train=True,download=True, transform=transforms.ToTensor())

        test_dataset = datasets.CIFAR10(root='./data',
        train=False,download=True, transform=transforms.ToTensor())

    else:
        sys.exit('dataset {} unknown. Select from {}'.format(args.dataset, default_datasets))

    
    filter_class_labels(train_dataset)
    filter_class_labels(test_dataset)
    num_train = len(train_dataset.train_labels)

    return train_dataset, test_dataset, num_train

def main(args):
    
    #import pdb; pdb.set_trace()
    train_dataset, test_dataset, num_train = get_datasets(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    reg = 1/num_train 

    losses_gn = []
    losses_gn_sketch = []
    losses_gn_half_sketch = []
    losses_sgd = []
    
    """ 
    print('Training using Gauss-Newton Solver')
    #Train using Gauss-Newton solver
    #model = Model(input_size=784, two_layer=args.two_layer) #init model
    #model = models.vgg11_bn(pretrained=True)
    model = VGG()
    optimizer = GN_Solver(model.parameters(), lr=1.0, reg=reg, backtrack=0)
    time_start = time.time()
    while optimizer.grad_update < args.max_iter:
        loss, accuracy = train_GN(model, train_dataloader, optimizer, losses_gn)
    time_end = time.time()
    t_solve_gn = time_end - time_start 
    print('Trained in {0:.3f}s'.format(t_solve_gn))
    loss, accuracy = test(model, test_dataloader)
    print('Test Loss: {}, Accuracy: {}'.format(loss, accuracy))

    print('-'*30)
    """
    #Train using Gauss-Newton Half-sketch
    print('Training using Gauss-Newton Half-Sketch Solver')
    #model = Model(input_size=784, two_layer=args.two_layer) #init model
    #model = models.vgg11_bn(pretrained=True)
    model = VGG()
    optimizer = GN_Solver(model.parameters(), lr=1.0, reg=reg, backtrack=0,
    sketch_size=784, max_iter=10)
    train_dataloader = DataLoader(train_dataset, batch_size=args.sketch_size, shuffle=True)
    time_start = time.time()
    while optimizer.grad_update < args.max_iter:
        loss, accuracy = train_GN(model, train_dataloader, optimizer, losses_gn_half_sketch)
    time_end = time.time()
    t_solve_gn_half_sketch = time_end - time_start 
    print('Trained in {0:.3f}s'.format(t_solve_gn_half_sketch))
    loss, accuracy = test(model, test_dataloader)
    print('Test Loss: {}, Accuracy: {}'.format(loss, accuracy))

    print('-'*30)
    
    print('Training using Gauss-Newton Sketch Solver')
    #Train using Gauss-Newton Sketch 
    #The sketch will be just a sample of the data, so we'll opt to use a mini-batch of sketch size
    train_dataloader = DataLoader(train_dataset, batch_size=args.sketch_size, shuffle=True)
    #model = Model(input_size=784, two_layer=args.two_layer) #init model
    #model = models.vgg11_bn(pretrained=True)
    model = VGG()
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = GN_Solver(model.parameters(), lr=1.0, reg=reg, backtrack=0,
    max_iter=10)
    time_start = time.time()
    while optimizer.grad_update < args.max_iter:
        loss, accuracy = train_GN(model, train_dataloader, optimizer, losses_gn_sketch)
    time_end = time.time()
    t_solve_gn_sketch = time_end - time_start 
    print('Trained in {0:.3f}s'.format(t_solve_gn_sketch))
    loss, accuracy = test(model, test_dataloader)
    print('Test Loss: {}, Accuracy: {}'.format(loss, accuracy))

    #Train using SGD
    print('Training using SGD')
    #model = Model(input_size=784, two_layer=args.two_layer) #re-init model 
    #model = models.vgg11_bn(pretrained=True)
    model = VGG()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=reg) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.sketch_size, shuffle=True)
    time_start = time.time()
    num_iter_updates = [0] #wrapped in list b/c integers are immutable
    while num_iter_updates[0] < 30:
        loss, accuracy = train_SGD(model, train_dataloader, optimizer, losses_sgd, num_iter_updates)

    time_end = time.time()
    t_solve_sgd = time_end - time_start 
    print('Trained in {0:.3f}s'.format(t_solve_sgd))
    loss, accuracy = test(model, test_dataloader)
    print('Test Loss: {}, Accuracy: {}'.format(loss, accuracy))

    """
    plt.semilogy(np.arange(len(losses_gn)) * t_solve_gn / len(losses_gn), losses_gn, 'k', 
                 np.arange(len(losses_gn_sketch)) * t_solve_gn_sketch / len(losses_gn_sketch), losses_gn_sketch, 'g', 
                 np.arange(len(losses_gn_half_sketch)) * t_solve_gn_half_sketch / len(losses_gn_half_sketch), losses_gn_half_sketch, 'r', 
                 np.arange(len(losses_sgd)) * t_solve_sgd / len(losses_sgd), losses_sgd, 'b')
    """
    plt.semilogy(np.arange(len(losses_gn_sketch)) * t_solve_gn_sketch / len(losses_gn_sketch), losses_gn_sketch, 'g', 
                 np.arange(len(losses_gn_half_sketch)) * t_solve_gn_half_sketch / len(losses_gn_half_sketch), losses_gn_half_sketch, 'r', 
                 np.arange(len(losses_sgd)) * t_solve_sgd / len(losses_sgd), losses_sgd, 'b')
    plt.title('Training loss')
    plt.xlabel('Seconds')
    plt.grid(True, which='both')
    plt.legend(['Gauss-Newton', 'Gauss-Newton Sketch', 'Gauss-Newton Half-Sketch', 'SGD'])
    plt.show()

def train_GN(model, dataloader, optimizer, all_losses):
   losses = []
   accuracy = []
   
   for idx, data in enumerate(dataloader):

       if optimizer.grad_update >= args.max_iter: #max iteration termination criteria
           break

       images, labels = data
       images = images.view(-1, 1, 28, 28) #reshape image to vector
       labels = labels.float()

       #Custom function b/c cost may need to be evaluated several times for backtracking
       def closure():
           optimizer.zero_grad()
           pred = model(images)
           loss = 0.5*F.mse_loss(pred.squeeze(), labels)
           #loss = 0.5*F.cross_entropy(pred.squeeze(), labels)
           #print(pred.squeeze().shape, labels.shape)
           err = pred - labels.unsqueeze(1) 
           return loss, err, pred
       
       loss, pred = optimizer.step(closure)
       
       print(pred[:10].squeeze(), labels[:10])
       losses.append(loss.item())
       all_losses.append(loss.item())
       acc = torch.sum((pred.squeeze(1)>0.5).float() == labels).float()/len(labels)
       accuracy.append(acc.item())
       print('Iter: {}, Loss: {}, Accuracy: {}'.format(optimizer.grad_update,loss.item(),acc.item()))

   return np.mean(losses), np.mean(accuracy)

def train_SGD(model, dataloader, optimizer, all_losses, num_iter_updates):
   losses = []
   accuracy = []
   
   for idx, data in enumerate(dataloader):
       images, labels = data
       #images = images.view(-1, 28*28*1) #reshape image to vector
       images = images.view(-1, 1, 28, 28) #reshape image to vector
       labels = labels.float()
       
       optimizer.zero_grad()
       pred = model(images)
       loss = 0.5*F.mse_loss(pred.squeeze(), labels)
       #loss = 0.5*F.cross_entropy(pred.squeeze(), labels)
       loss.backward()
       optimizer.step()

       losses.append(loss.item())
       all_losses.append(loss.item())
       acc = torch.sum((pred.squeeze(1)>0.5).float() == labels).float()/len(labels)
       accuracy.append(acc.item())
       print('Iter: {}, Loss: {}, Accuracy: {}'.format(idx,loss.item(),acc.item()))
    
   num_iter_updates[0] = num_iter_updates[0] + 1
   return np.mean(losses), np.mean(accuracy)

#forward testing pass
def test(model, dataloader):
    model.eval()

    losses = []
    accuracy = []

    for idx, data in enumerate(dataloader):
        images, labels = data
        #images = images.view(-1, 28*28*1)
        labels = labels.float()

        pred = model(images).squeeze()
        #loss = 0.5*F.cross_entropy(pred.squeeze(), labels)

        loss = 0.5*F.mse_loss(pred, labels)
        
        losses.append(loss.item())
        acc = torch.sum((pred>0.5).float() == labels).float()/len(labels)
        accuracy.append(acc.item())

    return np.mean(losses), np.mean(accuracy)

if __name__ == "__main__":
    main(args)
