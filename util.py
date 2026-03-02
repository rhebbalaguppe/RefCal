from __future__ import print_function

import math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from libauc.optimizers import PESG
import json
import os

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_output_labels_and_conf(output, target, opt,topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        target = torch.tensor(target)
        output = torch.tensor(output)
        
        sm = torch.nn.Softmax(dim=1)
        output = sm(output)

        maxk = max(topk)
        batch_size = target.size(0)

        conf_list, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        conf_list = conf_list.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        incorrect = ~correct
        correct_pred_conf = conf_list[correct]
        incorrect_pred_conf = conf_list[incorrect]

        conf_dict = {'correct_conf':correct_pred_conf.tolist(),'incorrect_conf':incorrect_pred_conf.tolist()}

        if opt.perform_temp_scaling:
            second_name = '_ts_labels.json'
        else:
            second_name = "_labels.json"

        with open(os.path.join(opt.save_folder,opt.loss_function+second_name),'w') as outfile:
            json.dump(conf_dict,outfile)
        #conf_df = pd.DataFrame.from_dict(conf_dict)
        print("File saved at ",os.path.join(opt.save_folder,opt.loss_function+second_name))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model,criterion):

    if 'loss_function' not in opt:
        optimizer = optim.SGD(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        # optimizer = optim.Adagrad(model.parameters(),
        #                     lr=opt.learning_rate,
        #                     weight_decay=opt.weight_decay)
        return optimizer

    if opt.loss_function == "AUCM_loss":
        optimizer = PESG(model.parameters(),loss_fn=criterion.criterion,lr=opt.learning_rate,gamma=500,margin=1.0,weight_decay=opt.weight_decay, device='cuda')
    else:
        optimizer = optim.SGD(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam,loss_function,**kwargs):
    if loss_function == "CRL":
        return lam * criterion(pred, y_a,args=kwargs['opt'],idx=kwargs['iter_idx'],history=kwargs['history']) + (1 - lam) * criterion(pred, y_b,args=kwargs['opt'],idx=kwargs['iter_idx'],history=kwargs['history'])
    else:
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
