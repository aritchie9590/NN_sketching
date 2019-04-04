import os 
import argparse
import numpy as np 

import torch
import torch.nn as nn
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
parser.add_argument('--reg', default=0.0001, type=float, help='regularizer')

parser.add_argument('--cuda', action='store_true', help='use gpu')

parser.set_defaults(cuda=False)
parser.set_defaults(two_layer=False)
args = parser.parse_args()

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

        #Set-up feed-forward network as two layer or single layer
        self.ff_network = self.two_layer if two_layer else self.single_layer 

    def forward(self, x):
        out = self.ff_network(x)

        return out

def get_dataloader(args):
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        sys.exit('dataset {} unknown. Select from {}'.format(args.dataset, default_datasets))

    return train_dataloader, test_dataloader

def main(args):
    train_dataloader, test_dataloader = get_dataloader(args)

    model = Model(input_size=784, two_layer=args.two_layer)

    '''
    n = len(train_dataloader)
    reg = 10/n #from the matlab version, lambda is 10/m
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=reg) 
    '''

    #import pdb; pdb.set_trace()
    optimizer = GN_Solver(model.parameters(), reg=args.reg)

    for epoch in range(args.num_epochs):
        loss, accuracy = train(epoch, model, train_dataloader, optimizer)
        print('Epoch: {}, Loss: {}, Accuracy: {}'.format(epoch, loss.item(), accuracy.item()))
    
    loss, accuracy = test(model, test_dataloader)
    print('Test Loss: {}, Accuracy: {}'.format(loss, accuracy))

#forward training pass
def train(epoch, model, dataloader, optimizer):
   model.train() #set model to training mode (typically only matters with dropout & batchnorm) 
   criterion = nn.MSELoss() 

   losses = []
   accuracy = []
   
   for idx, data in enumerate(dataloader):
       images, gt_labels = data
       labels = (gt_labels == 0).float() #transform to binary classification between 0 and non-zero
       images = images.view(-1, 28*28*1) #reshape image to vector

       #Custom function b/c conjuage gradient needs to re-evaluate the function multiple times
       def closure():
           optimizer.zero_grad()
           pred = model(images)
           
           loss = criterion(pred, labels)
           #loss.backward(retain_graph=True, create_graph=True) #TODO: May have to subtract gradient manually so they don't accumulate

           err = pred - labels.unsqueeze(1) 
           return loss, err, pred
       
       loss, pred = optimizer.step(closure)

       losses.append(loss.item())
       acc = torch.sum((pred.squeeze(1)>0.5).float() == labels).float()/len(labels)
       accuracy.append(acc.item())

   return np.mean(losses), np.mean(accuracy)

#forward testing pass
def test(model, dataloader):
    model.eval()
    criterion = nn.MSELoss()

    losses = []
    accuracy = []

    for idx, data in enumerate(dataloader):
        images, labels = data
        labels = (labels == 0).float() #transform to binary classification between 0 and non-zero

        images = images.view(-1, 28*28*1)

        pred = model(images).squeeze(1)

        loss = criterion(pred, labels)
        
        losses.append(loss.item())
        acc = torch.sum((pred>0.5).float() == labels).float()/len(labels)
        accuracy.append(acc.item())

    return np.mean(losses), np.mean(accuracy)

if __name__ == "__main__":
    main(args)
