# taken from https://github.com/Jonathan-Pearce/calibration_library/blob/master/metrics.py

import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from scipy.special import softmax

class BrierScore():
    def __init__(self) -> None:
        pass

    def loss(self, outputs, targets):
        K = outputs.shape[1]
        one_hot = np.eye(K)[targets]
        probs = softmax(outputs, axis=1)
        return np.mean( np.sum( (probs - one_hot)**2 , axis=1) )


class CELoss(object):

    def compute_bin_boundaries(self, probabilities = np.array([])):

        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]


    def get_probabilities(self, output, labels, logits):
        #If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.accuracies = np.equal(self.predictions,labels)

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        #self.acc_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)


    def compute_bins(self, index = None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = (self.labels == index).astype("float")


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])

class MaxProbCELoss(CELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()

#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_score)

class MCELoss(MaxProbCELoss):
    
    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.max(self.bin_score)

#https://arxiv.org/abs/1905.11001
#Overconfidence Loss (Good in high risk applications where confident but wrong predictions can be especially harmful)
class OELoss(MaxProbCELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_conf * np.maximum(self.bin_conf-self.bin_acc,np.zeros(self.n_bins)))


#https://arxiv.org/abs/1904.01685
class SCELoss(CELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        sce = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins(i)
            sce += np.dot(self.bin_prop, self.bin_score)

        return sce/self.n_class

# Taken from: https://github.com/by-liu/MbLS/blob/main/calibrate/evaluation/metrics.py
class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

class TACELoss(CELoss):

    def loss(self, output, labels, threshold = 0.01, n_bins = 15, logits = True):
        tace = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().get_probabilities(output, labels, logits)
        self.probabilities[self.probabilities < threshold] = 0
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bin_boundaries(self.probabilities[:,i]) 
            super().compute_bins(i)
            tace += np.dot(self.bin_prop,self.bin_score)

        return tace/self.n_class

#create TACELoss with threshold fixed at 0
class ACELoss(TACELoss):

    def loss(self, output, labels, n_bins = 15, logits = True):
        return super().loss(output, labels, 0.0 , n_bins, logits)
