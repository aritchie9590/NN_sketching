#!/usr/bin/env python

#Dataloader for the CT Dataset
#Reads the csv files

import os
import csv

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CT_Dataset(Dataset):
	
    def __init__(self, csv_file, train=True):
        super().__init__()

        self.data_points = []
        self.loader = transforms.ToTensor()

        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                patient_id = row[0]
                feature_vector = row[1:384]
                target = row[-1]

                data_point = {'feature':feature_vector, 'target':target}                
                self.data_points.append(data_point)

	#The split is 80% training data and 20% validation data
        if train:
            num_to_select = int(len(self.data_points) * 0.8)
            self.data_points = self.data_points[:num_to_select]
        else:
            num_to_select = int(len(self.data_points) * 0.2)
            self.data_points = self.data_points[-1*num_to_select:]


        print('{} data points'.format(len(self.data_points)))
            
    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, index):
        datapoint = self.data_points[index]

        feature = datapoint['feature']
        target = datapoint['target']

        return self.loader(feature), self.loader(target)
