import os 
import argparse
import numpy as np 

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

#settings
parser.add_argument('--dataset', default='mnist', type=str, help='dataset(s) to use')
default_datasets = ['mnist']
parser.add_argument('--num_epochs', default=10, type=int, help='num of epochs to train')
parser.add_argument('--batch_size', default=60000, type=int, help='batch size, 60,000 assumes all training data') 

parser.add_argument('--cuda', action='store_true', help='use gpu')

parser.set_defaults(cuda=False)
args = parser.parse_args()

#simple feed-forward neural network
class Model(nn.Module):
    
    def __init__(self, input_size=784):
        super().__init__()

        self.fc_layer = nn.Linear(input_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        h = self.fc_layer(x)
        out = self.activation(h)

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

    model = Model(input_size=784)

    #TODO: Our optimizer comes a second-order method
    reg = 10/60000 #from the matlab version, lambda is 10/m
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=reg) 

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
       images, labels = data
       labels = (labels == 0).float() #transform to binary classification between 0 and non-zero
       
       optimizer.zero_grad()
       images = images.view(-1, 28*28*1) #reshape image to vector

       pred = model(images).squeeze(1)
       
       loss = criterion(pred, labels)
       loss.backward()
       optimizer.step()

       losses.append(loss.item())
       acc = torch.sum((pred>0.5).float() == labels).float()/len(labels)
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
