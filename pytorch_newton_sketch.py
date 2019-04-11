import os 
import argparse
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from GN_Solver import GN_Solver

parser = argparse.ArgumentParser()

#settings
parser.add_argument('--dataset', default='mnist', type=str, help='dataset(s) to use')
default_datasets = ['mnist']
parser.add_argument('--two_layer', action='store_true', help='add parameter to use two-layer architecture instead of single-layer')
parser.add_argument('--num_epochs', default=10, type=int, help='num of epochs to train')
parser.add_argument('--batch_size', default=60000, type=int, help='mini-batch size') 
#parser.add_argument('--reg', default=0.0001, type=float, help='regularizer')

parser.add_argument('--cuda', action='store_true', help='use gpu')

parser.set_defaults(cuda=False)
parser.set_defaults(two_layer=False)
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

#simple feed-forward neural network
class Model(nn.Module):
    
    def __init__(self, input_size=784, hidden_dim=200, output_dim=1, two_layer=False):
        super().__init__()

        self.single_layer = nn.Sequential(
                            nn.Linear(input_size, 1, bias=False),
                            nn.Sigmoid())

        '''
        self.two_layer = nn.Sequential(
                         nn.Linear(input_size, hidden_dim),
                         nn.Sigmoid(), #Or ReLU()
                         nn.Linear(hidden_dim, 1),
                         nn.Sigmoid())
        '''
        self.two_layer = None 

        '''
        def init_weights(self, m):
                 for param in m.parameters():
                     if param.ndimension() > 1:
                         nn.init.xavier_uniform(param)
        '''
        #Set-up feed-forward network as two layer or single layer
        #self.ff_network = self.two_layer if two_layer else self.single_layer 

    def forward(self, x):
        #out = self.ff_network(x)
        out = self.single_layer(x)

        return out

#We will only perform binary classification between two labels
def filter_class_labels(dataset):
    #Filter out all digits except these two
    l1 = 0
    l2 = 1
    
    if dataset.train:
        idx_l1 = dataset.train_labels == l1
        idx_l2 = dataset.train_labels == l2 
        
        idx = idx_l1 + idx_l2
        dataset.train_labels = dataset.train_labels[idx]
        dataset.train_data = dataset.train_data[idx]
    else:
        idx_l1 = dataset.test_labels == l1
        idx_l2 = dataset.test_labels == l2

        idx = idx_l1 + idx_l2
        dataset.test_labels = dataset.test_labels[idx]
        dataset.test_data = dataset.test_data[idx]

def get_dataloader(args):
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())

        filter_class_labels(train_dataset)
        filter_class_labels(test_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        sys.exit('dataset {} unknown. Select from {}'.format(args.dataset, default_datasets))

    num_train = len(train_dataset.train_labels)

    return train_dataloader, test_dataloader, num_train

def main(args):
    #import pdb; pdb.set_trace()
    train_dataloader, test_dataloader, num_train = get_dataloader(args)
    reg = 1/num_train 

    print('Training using Gauss-Newton Solver')
    #Train using Gauss-Newton solver
    model = Model(input_size=784, two_layer=args.two_layer) #init model

    optimizer = GN_Solver(model.parameters(), lr=1.0, reg=reg, backtrack=0)
    for epoch in range(args.num_epochs):
        loss, accuracy = train_GN(epoch, model, train_dataloader, optimizer)
        print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss.item(), accuracy.item()))

    loss, accuracy = test(model, test_dataloader)
    print('Test Loss: {}, Accuracy: {}'.format(loss, accuracy))

    #TODO:
    #Train using Gauss-Newton Sketch \& Half-sketch
    
    #Train using SGD
    print('-'*30)
    print('Training using SGD')
    model = Model(input_size=784, two_layer=args.two_layer) #re-init model 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=reg) 
    for epoch in range(args.num_epochs):
        loss, accuracy = train_SGD(epoch, model, train_dataloader, optimizer)
        print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss.item(), accuracy.item()))

    loss, accuracy = test(model, test_dataloader)
    print('Test Loss: {}, Accuracy: {}'.format(loss, accuracy))

#forward training pass
def train_GN(epoch, model, dataloader, optimizer):
   model.train() #set model to training mode (typically only matters with dropout & batchnorm) 

   losses = []
   accuracy = []
   
   for idx, data in enumerate(dataloader):
       images, labels = data
       images = images.view(-1, 28*28*1) #reshape image to vector
       labels = labels.float()
       
       #Custom function b/c cost may need to be evaluated several times for backtracking
       def closure():
           optimizer.zero_grad()
           pred = model(images)
           
           loss = 0.5*F.mse_loss(pred.squeeze(), labels)

           err = pred - labels.unsqueeze(1) 
           return loss, err, pred, epoch
       
       if epoch == 9:
           print('Stop')

       loss, pred = optimizer.step(closure)

       losses.append(loss.item())
       acc = torch.sum((pred.squeeze(1)>0.5).float() == labels).float()/len(labels)
       accuracy.append(acc.item())

   return np.mean(losses), np.mean(accuracy)

def train_SGD(epoch, model, dataloader, optimizer):
   model.train() #set model to training mode (typically only matters with dropout & batchnorm) 

   losses = []
   accuracy = []
   
   for idx, data in enumerate(dataloader):
       images, labels = data
       images = images.view(-1, 28*28*1) #reshape image to vector
       labels = labels.float()
       
       optimizer.zero_grad()
       pred = model(images)
       loss = 0.5*F.mse_loss(pred.squeeze(), labels)
       loss.backward()
       optimizer.step()

       losses.append(loss.item())
       acc = torch.sum((pred.squeeze(1)>0.5).float() == labels).float()/len(labels)
       accuracy.append(acc.item())

   return np.mean(losses), np.mean(accuracy)

#forward testing pass
def test(model, dataloader):
    model.eval()

    losses = []
    accuracy = []

    for idx, data in enumerate(dataloader):
        images, labels = data
        images = images.view(-1, 28*28*1)
        labels = labels.float()

        pred = model(images).squeeze()

        loss = 0.5*F.mse_loss(pred, labels)
        
        losses.append(loss.item())
        acc = torch.sum((pred>0.5).float() == labels).float()/len(labels)
        accuracy.append(acc.item())

    return np.mean(losses), np.mean(accuracy)

if __name__ == "__main__":
    main(args)
