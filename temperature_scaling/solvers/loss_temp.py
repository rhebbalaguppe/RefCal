import logging
import os, json, math

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

from .mmce import MMCE_weighted
from .flsd import FocalLossAdaptive

from utils.torch_helper import one_hot
from utils.constants import EPS

class AdaFocal(nn.Module):
    def __init__(self, args, device=None, **kwargs):            
        super(AdaFocal, self).__init__()
        self.args = args
        self.num_bins = args.num_bins
        self.lamda = args.adafocal_lambda
        self.gamma_initial = args.adafocal_gamma_initial
        self.switch_pt = args.adafocal_switch_pt
        self.gamma_max = args.adafocal_gamma_max
        self.gamma_min = args.adafocal_gamma_min
        self.update_gamma_every = args.update_gamma_every
        self.device = device
        # This initializes the bin_stats variable
        self.bin_stats = collections.defaultdict(dict)
        for bin_no in range(self.num_bins):
            self.bin_stats[bin_no]['lower_boundary'] = bin_no*(1/self.num_bins)
            self.bin_stats[bin_no]['upper_boundary'] = (bin_no+1)*(1/self.num_bins)
            self.bin_stats[bin_no]['gamma'] = self.gamma_initial

    # This function updates the bin statistics which are used by the Adafocal loss at every epoch.
    def update_bin_stats(self, val_adabin_dict):
        for bin_no in range(self.num_bins):
            # This is the Adafocal gamma update rule
            prev_gamma = self.bin_stats[bin_no]['gamma']
            exp_term = val_adabin_dict[bin_no]['calibration_gap']
            if prev_gamma > 0:
                next_gamma = prev_gamma * math.exp(self.lamda*exp_term)
            else:
                next_gamma = prev_gamma * math.exp(-self.lamda*exp_term)    
            # This switches between focal and inverse-focal loss when required.
            if abs(next_gamma) < self.switch_pt:
                if next_gamma > 0:
                    next_gamma = -self.switch_pt
                else:
                    next_gamma = self.switch_pt
            self.bin_stats[bin_no]['gamma'] = max(min(next_gamma, self.gamma_max), self.gamma_min) # gamma-clipping
            self.bin_stats[bin_no]['lower_boundary'] = val_adabin_dict[bin_no]['lower_bound']
            self.bin_stats[bin_no]['upper_boundary'] = val_adabin_dict[bin_no]['upper_bound']
        # This saves the "bin_stats" to a text file.
        save_file = os.path.join(self.args.save_path, "val_bin_stats.txt")
        with open(save_file, "a") as write_file:
            json.dump(self.bin_stats, write_file)
            write_file.write("\n")
        return

    # This function selects the gammas for each sample based on which bin it falls into.
    def get_gamma_per_sample(self, pt):
        gamma_list = []
        batch_size = pt.shape[0]
        for i in range(batch_size):
            pt_sample = pt[i].item()
            for bin_no, stats in self.bin_stats.items():
                if bin_no==0 and pt_sample < stats['upper_boundary']:
                    break
                elif bin_no==self.num_bins-1 and pt_sample >= stats['lower_boundary']:
                    break
                elif pt_sample >= stats['lower_boundary'] and pt_sample < stats['upper_boundary']:
                    break
            gamma_list.append(stats['gamma'])
        return torch.tensor(gamma_list).to(self.device)

    # This computes the loss value to be returned for back-propagation.
    def forward(self, input, target, temp=1.0):
        if input.dim() > 2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        gamma = self.get_gamma_per_sample(pt)
        gamma_sign = torch.sign(gamma).cuda()
        gamma_mag = torch.abs(gamma).cuda()
        pt = gamma_sign * pt
        loss = -1 * ((1 - pt + 1e-20)**gamma_mag) * logpt # 1e-20 added for numerical stability 
 
        return temp * loss.sum()

# from https://github.com/torrvision/focal_calibration/blob/main/Losses/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, **kwargs):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        logging.info("using gamma={}".format(gamma))

    def forward(self, input, target, temp=1.0):

        target = target.view(-1,1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1-pt)**self.gamma * logpt
        
        return temp * loss.mean()

class CrossEntropy(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        logging.info("using cross entropy loss")

    def forward(self, input, target, temp=1.0):
        return temp * self.criterion(input, target)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, alpha=0.0, dim=-1, **kwargs):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - alpha
        self.alpha = alpha
        self.dim = dim

    def forward(self, pred, target, temp=1.0):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.shape[self.dim]
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.alpha / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return temp * torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class MDCA(torch.nn.Module):
    def __init__(self):
        super(MDCA,self).__init__()

    def forward(self , output, target):
        output = torch.softmax(output, dim=1)
        # [batch, classes]
        loss = torch.tensor(0.0).cuda()
        batch, classes = output.shape
        for c in range(classes):
            avg_count = (target == c).float().mean()
            avg_conf = torch.mean(output[:,c])
            loss += torch.abs(avg_conf - avg_count)
        denom = classes
        loss /= denom
        return loss

class ClassficationAndMDCA(nn.Module):
    def __init__(self, loss="NLL+MDCA", beta=1.0, gamma=1.0, **kwargs):
        super(ClassficationAndMDCA, self).__init__()
        self.loss = loss
        self.beta = beta
        self.gamma = gamma
        logging.info(f"using loss = {self.loss}")
        if "NLL" in loss:
            self.classification_loss = nn.CrossEntropyLoss()
            logging.info(f"using NLL + (beta={self.beta}) mdca")
        elif "FL" in loss:
            self.classification_loss = FocalLoss(gamma=self.gamma)
            logging.info(f"using FL (gamma={self.gamma}) + (beta={self.beta}) mdca")
        self.MDCA = MDCA()

    def forward(self, logits, targets, temp=1.0):
        loss_cls = self.classification_loss(logits, targets)
        loss_cal = self.MDCA(logits, targets)
        return temp * (loss_cls + self.beta * loss_cal)

class BrierScore(nn.Module):
    def __init__(self, **kwargs):
        super(BrierScore, self).__init__()

    def forward(self, logits, target):
        
        target = target.view(-1,1)
        target_one_hot = torch.FloatTensor(logits.shape).to(target.get_device())
        target_one_hot.zero_()
        target_one_hot.scatter_(1, target, 1)

        pt = torch.softmax(logits, dim=1)
        squared_diff = (target_one_hot - pt) ** 2

        loss = torch.sum(squared_diff) / float(logits.shape[0])
        return loss

class DCA(nn.Module):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__()
        self.beta = beta
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        output = torch.softmax(logits, dim=1)
        conf, pred_labels = torch.max(output, dim = 1)
        calib_loss = torch.abs(conf.mean() -  (pred_labels == targets).float().mean())
        return self.cls_loss(logits, targets) + self.beta * calib_loss

class MMCE(nn.Module):
    def __init__(self, beta=2.0, **kwargs):
        super().__init__()
        self.beta = beta
        self.mmce = MMCE_weighted()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, temp):
        cls = self.cls_loss(logits, targets)
        calib = self.mmce(logits, targets)
        return temp * (cls + self.beta * calib)

class FLSD(nn.Module):
    def __init__(self, gamma=3.0, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.criterion = FocalLossAdaptive(gamma=self.gamma)

    def forward(self, logits, targets):
        return self.criterion.forward(logits, targets)


class MbLS(nn.Module):
    """Add marginal penalty to logits:
        CE + alpha * max(0, max(l^n) - l^n - margin)

    Args:
        margin (float, optional): The margin value. Defaults to 10.
        alpha (float, optional): The balancing weight. Defaults to 0.1.
        ignore_index (int, optional):
            Specifies a target value that is ignored
            during training. Defaults to -100.

        The following args are related to balancing weight (alpha) scheduling.
        Note all the results presented in our paper are obtained without the scheduling strategy.
        So it's fine to ignore if you don't want to try it.

        schedule (str, optional):
            Different stragety to schedule the balancing weight alpha or not:
            "" | add | multiply | step. Defaults to "" (no scheduling).
            To activate schedule, you should call function
            `schedula_alpha` every epoch in your training code.
        mu (float, optional): scheduling weight. Defaults to 0.
        max_alpha (float, optional): Defaults to 100.0.
        step_size (int, optional): The step size for updating alpha. Defaults to 100.
    """
    def __init__(self,
                 margin: float = 10,
                 alpha: float = 0.1,
                 ignore_index: int = -100,
                 schedule: str = "",
                 mu: float = 0,
                 max_alpha: float = 100.0,
                 step_size: int = 100, **kwargs):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.mu = mu
        self.schedule = schedule
        self.max_alpha = max_alpha
        self.step_size = step_size

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_margin_l1"

    def schedule_alpha(self, epoch):
        """Should be called in the training pipeline if you want to se schedule alpha
        """
        if self.schedule == "add":
            self.alpha = min(self.alpha + self.mu, self.max_alpha)
        elif self.schedule == "multiply":
            self.alpha = min(self.alpha * self.mu, self.max_alpha)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.alpha = min(self.alpha * self.mu, self.max_alpha)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets, temp):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)
        # get logit distance
        diff = self.get_diff(inputs)
        # linear penalty where logit distances are larger than the margin
        loss_margin = F.relu(diff-self.margin).mean()
        loss = temp * (loss_ce + self.alpha * loss_margin)

        return loss, loss_ce, loss_margin

# implementation of CPC loss 
# Reference:
#   Calibrating Deep Neural Networks by Pairwise Constraints. CVPR 2022


class CPC(nn.Module):
    def __init__(self, lambd_bdc=1.0, lambd_bec=1.0, ignore_index=-100, **kwargs):
        super().__init__()
        self.lambd_bdc = lambd_bdc
        self.lambd_bec = lambd_bec
        self.ignore_index = ignore_index

        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_bdc", "loss_bec"

    def bdc(self, logits, targets_one_hot):
        # 1v1 Binary Discrimination Constraints (BDC)
        logits_y = logits[targets_one_hot == 1].view(logits.size(0), -1)
        logits_rest = logits[targets_one_hot == 0].view(logits.size(0), -1)
        loss_bdc = - F.logsigmoid(logits_y - logits_rest).sum() / (logits.size(1) - 1) / logits.size(0)

        return loss_bdc

    def bec(self, logits, targets_one_hot):
        # Binary Exclusion COnstraints (BEC)
        logits_rest = logits[targets_one_hot == 0].view(logits.size(0), -1)
        diff = logits_rest.unsqueeze(2) - logits_rest.unsqueeze(1)
        loss_bec = - torch.sum(
            0.5 * F.logsigmoid(diff + EPS)
            / (logits.size(1) - 1) / (logits.size(1) - 2) / logits.size(0)
        )

        return loss_bec

    def forward(self, inputs, targets, temp=1.0):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        loss_ce = self.cross_entropy(inputs, targets)

        targets_one_hot = one_hot(targets, inputs.size(1))
        loss_bdc = self.bdc(inputs, targets_one_hot)
        loss_bec = self.bec(inputs, targets_one_hot)

        loss = loss_ce + self.lambd_bdc * loss_bdc + self.lambd_bec * loss_bec

        return temp * loss, loss_ce, loss_bdc, loss_bec


class ACLSLoss(nn.Module):
    ## M is set to 6 for CIFAR-10 and 10 for other datasets. For CIFAR-100 as well, we are using 6.
    def __init__(self, lambda1=0.1, lambda2=0.01, M=6, dim=-1, **kwargs):
        super(ACLSLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.M = M
        self.dim = dim

    def forward(self, logits, target, temp=1.0):
        pred = logits.log_softmax(dim=self.dim)
        num_classes = pred.shape[self.dim]

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            target_one_hot = F.one_hot(target, num_classes=num_classes)
            min_z_k = torch.min(logits, dim=self.dim, keepdim=True)[0]
            z_y = logits.gather(self.dim, target.unsqueeze(self.dim))  # Logit corresponding to the target class
            #true_dist.fill_(self.lambda2 * F.relu(z_y - logits - self.M))
            # Compute the ACLS loss for j = y^
            true_dist.scatter_(self.dim, target.unsqueeze(self.dim), (1 - self.lambda1 * F.relu(z_y - min_z_k - self.M)))
            # Compute the ACKS loss for j != y^
            true_dist_other = (1 - target_one_hot).squeeze(1) * self.lambda2 * F.relu(z_y - logits - self.M)
            true_dist_other.scatter_(self.dim, target.unsqueeze(self.dim), 0)
            true_dist += true_dist_other

        return temp * torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


#lass LabelSmoothingLoss(nn.Module):
#   def __init__(self, alpha=0.0, dim=-1, **kwargs):
#       super(LabelSmoothingLoss, self).__init__()
#       self.confidence = 1.0 - alpha
#       self.alpha = alpha
#       self.dim = dim
#
#   def forward(self, pred, target, temp=1.0):
#       pred = pred.log_softmax(dim=self.dim)
#       num_classes = pred.shape[self.dim]
#       with torch.no_grad():
#           # true_dist = pred.data.clone()
#            true_dist = torch.zeros_like(pred)
#            true_dist.fill_(self.alpha / (num_classes - 1))
#            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#        return temp * torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

loss_dict = {
    "cross_entropy" : CrossEntropy,
    # "mdca" : ClassficationAndMDCA,
    "NLL+MDCA" : ClassficationAndMDCA,
    "FL+MDCA" : ClassficationAndMDCA,
    "focal_loss" : FocalLoss,
    "LS" : LabelSmoothingLoss,
    "adafocal" : AdaFocal,
    "MbLS" : MbLS,
    "CPC" : CPC,
    "MMCE" : MMCE,
    "ACLS" : ACLSLoss
}
