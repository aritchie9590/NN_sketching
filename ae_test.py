import sys
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
from CT_Dataset import CT_Dataset

parser = argparse.ArgumentParser()

#settings
parser.add_argument('--dataset', default='mnist', type=str, help='dataset(s) to use')
default_datasets = ['mnist', 'cifar10', 'ct']
parser.add_argument('--two_layer', action='store_true', help='add parameter to use two-layer architecture instead of single-layer')
parser.add_argument('--max_iter', default=80, type=int, help='max num of iterations to train on')
parser.add_argument('--batch_size', default=60000, type=int, help='mini-batch size') 

parser.add_argument('--sketch_size', default=700, type=int, help='sketch size for sketching algorithms')
#parser.add_argument('--reg', default=0.0001, type=float, help='regularizer')

parser.add_argument('--cuda', action='store_true', help='use gpu')

parser.set_defaults(cuda=False)
#parser.set_defaults(two_layer=False)
parser.set_defaults(two_layer=True)
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)
device = 'cuda' if args.cuda else 'cpu'

#simple feed-forward neural network
class Model(nn.Module):
    
    def __init__(self, input_size=784, hidden_dim=200, output_dim=1, two_layer=False):
        super().__init__()

        self.ae = nn.Sequential(
                         nn.Linear(input_size, 1000),
                         nn.ReLU(), #Or ReLU()
                         nn.Linear(1000, 500),
                         nn.ReLU(),
                         nn.Linear(500, 250),
                         nn.ReLU(),
                         nn.Linear(250, 30),
                         nn.ReLU(),
                         nn.Linear(30, 250),
                         nn.ReLU(),
                         nn.Linear(250, 500),
                         nn.ReLU(),
                         nn.Linear(500, 1000),
                         nn.ReLU(),
                         nn.Linear(1000, input_size),
                         )

    def forward(self, x):
        out = self.ae(x)
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

def get_datasets(args):
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data/', train=True,
        download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root='./data/', train=False,
        download=True, transform=transforms.ToTensor())

        num_train = len(train_dataset.train_labels)
    elif args.dataset == 'cifar10':
        
        train_dataset = datasets.CIFAR10(root='./data',
        train=True,download=True,
        transform=transforms.Compose([transforms.CenterCrop(28),
        transforms.ToTensor()])
        )

        test_dataset = datasets.CIFAR10(root='./data',
        train=False,download=True,
        transform=transforms.Compose([transforms.CenterCrop(28),
        transforms.ToTensor()])
        )
        
        num_train = len(train_dataset.train_labels)
    elif args.dataset == 'ct':
        train_dataset = CT_Dataset('./data/slice_localization_data.csv', train=True)
        test_dataset = CT_Dataset('./data/slice_localization_data.csv', train=False)

        num_train = len(train_dataset)
    else:
        sys.exit('dataset {} unknown. Select from {}'.format(args.dataset, default_datasets))
    
    #filter_class_labels(train_dataset)
    #filter_class_labels(test_dataset)
    return train_dataset, test_dataset, num_train

def main(args):
    
    #import pdb; pdb.set_trace()
    train_dataset, test_dataset, num_train = get_datasets(args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    reg = 100/num_train 
    #reg = 0

    losses_gn = []
    losses_gn_sketch = []
    losses_gn_half_sketch = []
    losses_sgd = []
    losses_adam = []
    vlosses_gn = []
    vlosses_gn_sketch = []
    vlosses_gn_half_sketch = []
    vlosses_sgd = []
    vlosses_adam = []

    losses_gni_val = []
    losses_gn_sketch_val = []
    losses_gn_half_sketch_val = []
    losses_sgd_val = []
    losses_adam_val = []

    input_size=784
    output_dim = 784

    """ 
    print('Training using Gauss-Newton Solver')
    #Train using Gauss-Newton solver
    #model = Model(input_size=input_size, two_layer=args.two_layer,output_dim=output_dim).to(device) #init model
    #model = models.vgg11_bn(pretrained=True).to(device)
    model = VGG().to(device)
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
    model = Model(input_size=input_size, two_layer=args.two_layer,output_dim=output_dim).to(device) #init model
    #model = models.vgg11_bn(pretrained=True).to(device)
    #model = VGG().to(device)
    optimizer = GN_Solver(model.parameters(), lr=1.0, reg=reg, backtrack=0,
    max_iter=10)
    train_dataloader = DataLoader(train_dataset, batch_size=args.sketch_size, shuffle=True)
    time_start = time.time()
    while optimizer.grad_update < args.max_iter:
        loss, accuracy = train_GN(model, train_dataloader, test_dataloader,
        optimizer, losses_gn_half_sketch, vlosses_gn_half_sketch)

    time_end = time.time()
    t_solve_gn_half_sketch = time_end - time_start 
    print('Trained in {0:.3f}s'.format(t_solve_gn_half_sketch))

    print('-'*30)
    
    print('Training using Gauss-Newton Sketch Solver')
    #Train using Gauss-Newton Sketch 
    #The sketch will be just a sample of the data, so we'll opt to use a mini-batch of sketch size
    train_dataloader = DataLoader(train_dataset, batch_size=args.sketch_size, shuffle=True)
    model = Model(input_size=input_size, two_layer=args.two_layer, output_dim=output_dim).to(device) #init model
    #model = models.vgg11_bn(pretrained=True).to(device)
    #model = VGG().to(device)
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = GN_Solver(model.parameters(), lr=1.0, reg=reg, backtrack=0,
    max_iter=10)
    time_start = time.time()
    while optimizer.grad_update < args.max_iter:
        loss, accuracy = train_GN(model, train_dataloader, optimizer, losses_gn_sketch)

        loss, accuracy = train_GN(model, train_dataloader, test_dataloader,
        optimizer, losses_gn_sketch, vlosses_gn_sketch)
    time_end = time.time()
    t_solve_gn_sketch = time_end - time_start 
    print('Trained in {0:.3f}s'.format(t_solve_gn_sketch))

    #Train using SGD
    print('Training using SGD')
    model = Model(input_size=input_size, two_layer=args.two_layer, output_dim=output_dim).to(device) #re-init model 
    #model = models.vgg11_bn(pretrained=True).to(device)
    #model = VGG().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=reg) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.sketch_size, shuffle=True)
    time_start = time.time()
    num_iter_updates = [0] #wrapped in list b/c integers are immutable
    while num_iter_updates[0] < 4:
        loss, accuracy = train_SGD(model, train_dataloader, test_dataloader,
        optimizer, losses_sgd, num_iter_updates, vlosses_sgd)

    time_end = time.time()
    t_solve_sgd = time_end - time_start 
    print('Trained in {0:.3f}s'.format(t_solve_sgd))
    
    #Train using ADAM
    print('Training using ADAM')
    model = Model(input_size=input_size, two_layer=args.two_layer, output_dim=output_dim).to(device) #re-init model 
    #model = models.vgg11_bn(pretrained=True).to(device)
    #model = VGG().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=reg) 
    train_dataloader = DataLoader(train_dataset, batch_size=args.sketch_size, shuffle=True)
    time_start = time.time()
    num_iter_updates = [0] #wrapped in list b/c integers are immutable
    while num_iter_updates[0] < 4:
        loss, accuracy = train_SGD(model, train_dataloader, test_dataloader, optimizer,
        losses_adam, num_iter_updates, vlosses_adam)

    time_end = time.time()
    t_solve_adam = time_end - time_start 
    print('Trained in {0:.3f}s'.format(t_solve_adam))
    
    """
    plt.semilogy(np.arange(len(losses_gn)) * t_solve_gn / len(losses_gn), losses_gn, 'k', 
                 np.arange(len(losses_gn_sketch)) * t_solve_gn_sketch / len(losses_gn_sketch), losses_gn_sketch, 'g', 
                 np.arange(len(losses_gn_half_sketch)) * t_solve_gn_half_sketch / len(losses_gn_half_sketch), losses_gn_half_sketch, 'r', 
                 np.arange(len(losses_sgd)) * t_solve_sgd / len(losses_sgd), losses_sgd, 'b')
    """
    #plot training error
    plt.subplot(2, 1, 2)
    plt.semilogy(np.arange(len(losses_gn_sketch)) * t_solve_gn_sketch / len(losses_gn_sketch), losses_gn_sketch, 'g', 
                 np.arange(len(losses_gn_half_sketch)) * t_solve_gn_half_sketch / len(losses_gn_half_sketch), losses_gn_half_sketch, 'r', 
                 np.arange(len(losses_sgd)) * t_solve_sgd / len(losses_sgd),losses_sgd, 'b',
                 np.arange(len(losses_adam)) * t_solve_adam / len(losses_adam), losses_adam, 'c')
    plt.title('Training loss')
    plt.xlabel('Seconds')
    plt.grid(True, which='both')
    #plt.legend(['Gauss-Newton', 'Gauss-Newton Sketch', 'Gauss-Newton Half-Sketch', 'SGD'])
    plt.legend(['Gauss-Newton Sketch', 'Gauss-Newton Half-Sketch', 'SGD', 'ADAM'])

    plt.plot(np.arange(len(vlosses_gn_sketch)) * t_solve_gn_sketch /len(vlosses_gn_sketch), vlosses_gn_sketch, 'g', 
                 np.arange(len(vlosses_gn_half_sketch)) * t_solve_gn_half_sketch / len(vlosses_gn_half_sketch), vlosses_gn_half_sketch, 'r', 
                 np.arange(len(vlosses_sgd)) * t_solve_sgd / len(vlosses_sgd),vlosses_sgd, 'b',
                 np.arange(len(vlosses_adam)) * t_solve_adam / len(vlosses_adam), vlosses_adam, 'c')
    plt.title('Validation Accuracy')
    plt.xlabel('Seconds')
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

def train_GN(model, dataloader, valLoader, optimizer, all_losses, val_losses):
   losses = []
   accuracy = []

   for idx, data in enumerate(dataloader):

       if optimizer.grad_update >= args.max_iter: #max iteration termination criteria
           break

       images, labels = data
       images = images.view(-1, 784);
       #images = images.view(-1, 1, 28, 28) #reshape image to vector
       labels = labels.float()
        
       it = iter(valLoader)
       vims, vlabels  = next(it)
       vims = vims.view(-1, 784)

       images = images.to(device)
       labels = labels.to(device)

       vims = vims.to(device)
       vlabels = vlabels.to(device)
       #Custom function b/c cost may need to be evaluated several times for backtracking
       def closure():
           optimizer.zero_grad()
           pred = model(images)

           loss = 0.5*F.mse_loss(pred, images)
           err = images - pred
           return loss, err, pred
       
       loss, pred = optimizer.step(closure)
       
       #print(pred[:10].squeeze(), labels[:10])
       losses.append(loss.item())
       all_losses.append(loss.item())
       val_preds = model(vims)
       vlosses = 0.5*F.mse_loss(model(vims), vims)
       val_losses.append(vlosses.item())
       print('Iter: {}, Loss: {}, V Loss:{}'.format(optimizer.grad_update,loss.item(),vlosses.item()))

   return np.mean(losses), np.mean(accuracy)

def train_SGD(model, dataloader, valLoader, optimizer, all_losses,
num_iter_updates, val_losses):
   losses = []
   accuracy = []
   
   it = iter(valLoader)
   vims, vlabels  = next(it)
   vims = vims.view(-1, 784)

   vims = vims.to(device)
   vlabels = vlabels.to(device)

   for idx, data in enumerate(dataloader):
       images, labels = data
       images = images.view(-1, 784)
       labels = labels.float()

       images = images.to(device)
       labels = labels.to(device)
       
       optimizer.zero_grad()
       pred = model(images)
       loss = 0.5*F.mse_loss(pred, labels)
       loss.backward()
       optimizer.step()

       losses.append(loss.item())
       all_losses.append(loss.item())

       val_preds = model(vims) 
       
       vlosses = 0.5*F.mse_loss(model(vims), vims)
       val_losses.append(vlosses.item())
       print('Iter: {}, Loss: {}, V Loss:{}'.format(optimizer.grad_update,loss.item(),vlosses.item()))
    
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
        
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        #loss = F.cross_entropy(pred, labels.type(torch.LongTensor))

        loss = 0.5*F.mse_loss(pred, labels)
        
        losses.append(loss.item())
        acc = torch.sum((pred.squeeze(1)>0.5).float() == labels).float()/len(labels)
        #acc = torch.sum((torch.argmax(pred, 1)).float() == labels).float()/len(labels) 
        accuracy.append(acc.item())

    return np.mean(losses), np.mean(accuracy)

if __name__ == "__main__":
    main(args)
