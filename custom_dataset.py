import os
import numpy as np
import torch
import torchvision as tv
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets

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

        return x, self.y_data[idx], idx
    
class Custom_Dataset_Supcon(Dataset):
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
    
class Custom_Dataset_stl10_binary(Dataset):
    def __init__(self, x, y, data_set, transform=None):
        self.x_data = x
        self.data = data_set
        self.transform = transform

        #write functions to modify the labels
        class_0_list = [0,2,8,9]
        class_1_list = [1,3,4,5,6,7]
        y_temp = []
        for t in y:
            if t in class_0_list:
                y_temp.append(0)
            elif t in class_1_list:
                y_temp.append(1)
            else:
                print("Errrrrrror")

        self.y_data = y_temp
        self.targets = y_temp

    def __len__(self):
        return len(self.x_data)

    # return idx
    def __getitem__(self, idx):
        if self.data == 'cifar' or self.data == 'imagenet':
            img = Image.fromarray(self.x_data[idx])
        elif self.data == 'stl10_binary':
            img = Image.fromarray(np.transpose(self.x_data[idx], (1, 2, 0)))

        x = self.transform(img)

        return x, self.y_data[idx], idx

class Custom_Dataset_Imagenet(Dataset):
    def __init__(self, images, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        for x in images:
            self.img_path.append(x[0])
            self.labels.append(int(x[1]))

        self.targets = self.labels

    def __len__(self):
        return len(self.labels)

    # return idx
    def __getitem__(self, idx):

        path = self.img_path[idx]
        label = self.labels[idx]
        
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        x = self.transform(img)

        return x, label, idx

class Custom_Dataset_Imagenet_Supcon(Dataset):
    def __init__(self, images, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        for x in images:
            self.img_path.append(x[0])
            self.labels.append(int(x[1]))

    def __len__(self):
        return len(self.labels)

    # return idx
    def __getitem__(self, idx):

        path = self.img_path[idx]
        label = self.labels[idx]
        
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        x = self.transform(img)

        return x, label
    
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

def load_imagenet_lt(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True,transform=None):
    
    txt = os.path.join(data_root,"imagenet/ImageNet_LT_%s.txt"%((phase)))

    print('Loading data from %s' % (txt))
    print('Use data transformation:', transform)

    set_ = LT_Dataset(os.path.join(data_root,'imagenet/ILSVRC/Data/CLS-LOC'), txt, transform)

    return set_

class LT_Dataset_non_supcon(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label,index

def load_imagenet_lt_non_supcon(data_root, dataset, phase, batch_size, sampler_dic=None, num_workers=4, test_open=False, shuffle=True,transform=None):
    
    txt = os.path.join(data_root,"imagenet/ImageNet_LT_%s.txt"%((phase)))

    print('Loading data from %s' % (txt))
    print('Use data transformation:', transform)

    set_ = LT_Dataset_non_supcon(os.path.join(data_root,'imagenet/ILSVRC/Data/CLS-LOC'), txt, transform)

    return set_