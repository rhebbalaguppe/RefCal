# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
from .calData import testData,testUni,testGaussian
from .calMetric import metric
from PIL import Image
from torch.utils.data import Dataset

#from ..custom_dataset import Custom_Dataset

class Custom_Dataset(Dataset):
    def __init__(self, x, y, data_set, transform=None):
        self.x_data = x
        self.y_data = y
        self.targets = y
        self.data = data_set
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    # return idx
    def __getitem__(self, idx):
        if self.data == 'cifar':
            img = Image.fromarray(self.x_data[idx])
        elif self.data == 'c2':
            img = Image.fromarray(self.x_data[idx].astype(np.uint8))
        elif self.data == 'svhn':
            img = Image.fromarray(np.transpose(self.x_data[idx], (1, 2, 0)))

        x = self.transform(img)

        return x, self.y_data[idx]

# import .calData as d
#CUDA_DEVICE = 0

start = time.time()
#loading data sets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])




# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
#nnName = "densenet10"

#imName = "Imagenet"



criterion = nn.CrossEntropyLoss()



def test(nnName, dataName, CUDA_DEVICE, epsilon, temperature,net1,test_criterion,classifier=None,opt=None):
    
    optimizer1 = optim.SGD(net1.parameters(), lr = 0, momentum = 0)
    #net1.cuda(CUDA_DEVICE)
    
    if dataName != "Uniform" and dataName != "Gaussian":
        testsetout = torchvision.datasets.SVHN("./datasets",split='test',transform=transform,download=True)
        testsetout_final = Custom_Dataset(testsetout.data,testsetout.labels,'svhn', transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout_final, batch_size=1,shuffle=False, num_workers=2)

    if nnName == "resnet50":
        testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
        testset_final = Custom_Dataset(testset.data,testset.targets,'cifar', transform)
        testloaderIn = torch.utils.data.DataLoader(testset_final, batch_size=1,shuffle=False, num_workers=2)
    
    if nnName == "densenet100" or nnName == "wideresnet100":
        testset = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False, num_workers=2)
    
    if dataName == "Gaussian":
        testGaussian(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, nnName, dataName, epsilon, temperature)
        metric(nnName, dataName)

    elif dataName == "Uniform":
        testUni(net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, nnName, dataName, epsilon, temperature)
        metric(nnName, dataName)
    else:
        f1,f2 = testData(net1, test_criterion, CUDA_DEVICE, testloaderIn, testloaderOut, nnName, dataName, epsilon, temperature,classifier=classifier,opt=opt)
        metric(nnName, dataName,f1,f2)








