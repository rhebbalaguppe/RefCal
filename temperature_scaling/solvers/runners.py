import torch
import torch.nn.functional as F

import sys
import numpy

from tqdm import tqdm
from utils import AverageMeter, accuracy, get_lr

import solvers.crl_utils

import numpy as np
from calibration_library.metrics import ECELoss, SCELoss, AdaptiveECELoss

from sklearn.metrics import roc_auc_score

from torchmetrics import AUROC

from transformers.modeling_outputs import SequenceClassifierOutput

from accelerate import Accelerator

from .mixup import mixup_data, mixup_criterion
from torch.autograd import Variable

numpy.set_printoptions(threshold=sys.maxsize)

def train(trainloader, model, optimizer, criterion, scheduler, accelerator:Accelerator, curr_epoch, args):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    bar = tqdm(enumerate(trainloader), total=len(trainloader), disable=(not accelerator.is_main_process))
    for batch_idx, (inputs, targets) in bar:
        if args.mixup:
            # generate mixed inputs, two one-hot label vectors and mixing coefficient
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup_alpha, True, args)
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            # inputs, targets = inputs.cuda(), targets.cuda()
            # compute output
            #inputs = torch.tensor(inputs).cuda().long()
            if args.dataset == "20_newsgroup":
                inputs = inputs.cuda()
            else:
                inputs = inputs.cuda()
            #print(inputs)
            outputs = model(inputs)
        else:
            # inputs, targets = inputs.cuda(), targets.cuda()
            # compute output
            if "bert" in args.model or "bert" in args.teacher:
                inputs['input_ids'] = inputs['input_ids'].squeeze(1)
                outputs = model(inputs['input_ids'])
            else:
                outputs = model(inputs)
        
        if isinstance(outputs, SequenceClassifierOutput):
            outputs = outputs.logits
        
        if args.mixup:
            mixup_loss = mixup_criterion(targets_a, targets_b, lam, args.temp)
            loss = mixup_loss(criterion, outputs)
        elif args.loss == "MbLS":
            loss, _, _ = criterion(outputs, targets, args.temp) #args.temp)
        elif args.loss == "CPC":
            loss, _, _, _ = criterion(outputs, targets, args.temp)
        else:
            # Change args.temp as 1 if u want to use the loss functions normally
            loss = criterion(outputs, targets, args.temp) #args.temp)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        if args.scheduler == "warmupcosine":
            scheduler.step()
        if args.scheduler == "timmwarmupcosine":
            scheduler.step(curr_epoch)
        
        if "bert" in args.model:
            if args.mixup:
               inputs_ids = inputs
            else:
               inputs_ids = inputs['input_ids']
        else:
            inputs_ids = inputs
        
        # measure accuracy and record loss for rank-0 only
        if accelerator.is_main_process:
            prec1, = accuracy(outputs.data, targets.data, topk=(1, ))
            top1.update(prec1.item(), inputs_ids.size(0))
            losses.update(loss.item(), inputs_ids.size(0))

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | lr {lr: .5f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    lr=get_lr(optimizer)
                    ))

    return (losses.avg, top1.avg)

def train_student(trainloader, student, teacher, optimizer, criterion, scheduler, accelerator:Accelerator, curr_epoch, args):
    # switch to train mode on student
    # eval mode on teacher
    student.train()
    teacher.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    bar = tqdm(enumerate(trainloader), total=len(trainloader), disable=(not accelerator.is_main_process))
    for batch_idx, (inputs, targets) in bar:
        # inputs, targets = inputs.cuda(), targets.cuda()
        if ("bert" in args.model) and (args.dataset == "20_newsgroup"):
            inputs['input_ids'] = inputs['input_ids'].squeeze(1)
            outputs = student(inputs['input_ids'])
            if isinstance(outputs, SequenceClassifierOutput):
                outputs = outputs.logits
        else:
            outputs = student(inputs)
        
        if accelerator.autocast():
           if ("bert" in args.teacher) and (args.dataset == "20_newsgroup"):
               inputs['input_ids'] = inputs['input_ids'].squeeze(1)
               with torch.no_grad():
                    outputs_teacher = teacher(inputs['input_ids'])
               if isinstance(outputs_teacher, SequenceClassifierOutput):
                   outputs_teacher = outputs_teacher.logits
               loss = criterion(outputs, outputs_teacher, targets)
           else:
               with torch.no_grad():
                    outputs_teacher = teacher(inputs)
               loss = criterion(outputs, outputs_teacher, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()

        if args.scheduler == "warmupcosine":
            scheduler.step()
        if args.scheduler == "timmwarmupcosine":
            scheduler.step(curr_epoch)

        if args.dataset == "20_newsgroup":
            if args.mixup:
               inputs_ids = inputs
            else:
               inputs_ids = inputs['input_ids']
        else:
            inputs_ids = inputs
        # measure accuracy and record loss
        if accelerator.is_main_process:
            prec1, = accuracy(outputs.data, targets.data, topk=(1, ))
            losses.update(loss.item(), inputs_ids.size(0))
            top1.update(prec1.item(), inputs_ids.size(0))

        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | lr {lr: .5f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    lr=get_lr(optimizer)
                    ))

    return (losses.avg, top1.avg)

@torch.no_grad()
def test(testloader, model, criterion, accelerator:Accelerator, args):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader), disable=(not accelerator.is_main_process))
    for batch_idx, (inputs, targets) in bar:
        #auroc_metric = AUROC(task='multiclass', num_classes=100)
        # inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        if "bert" in args.model or "bert" in args.teacher:
            inputs['input_ids'] = inputs['input_ids'].squeeze(1)
            outputs = model(inputs['input_ids'])
        else:
            outputs = model(inputs)
        
        if isinstance(outputs, SequenceClassifierOutput):
            outputs = outputs.logits
        
        loss = criterion(outputs, targets)
        
        prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))

        # gather metrics and outputs
        prec1, prec3, prec5, loss = accelerator.gather((prec1, prec3, prec5, loss))
        outputs, targets = accelerator.gather((outputs, targets))
        #print(outputs, targets)
        #print(targets.shape)
	    #auroc = auroc_metric(outputs, targets)
	    
        if "bert" in args.model or "bert" in args.teacher:
            inputs_ids = inputs['input_ids']
        else:
            inputs_ids = inputs
        
        if accelerator.is_main_process:
            losses.update(loss.mean().item(), inputs_ids.size(0))
            top1.update(prec1.mean().item(), inputs_ids.size(0))
            top3.update(prec3.mean().item(), inputs_ids.size(0))
            top5.update(prec5.mean().item(), inputs_ids.size(0))

            targets = targets.cpu().numpy()
            outputs = outputs.cpu().numpy()

            if all_targets is None:
                all_outputs = outputs
                all_targets = targets
            else:
                all_targets = np.concatenate([all_targets, targets], axis=0)
                all_outputs = np.concatenate([all_outputs, outputs], axis=0)
    
        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))
    
    if accelerator.is_main_process:
        ECE = ECELoss().loss(all_outputs, all_targets, n_bins=15)
        SCE = SCELoss().loss(all_outputs, all_targets, n_bins=15)
        #AECE_class = AdaptiveECELoss(n_bins=15)
        #AECE = AECE_class(all_outputs, all_targets)
    else:
        ECE = None
        SCE = None
        #auroc = None

    
    return (losses.avg, top1.avg, top3.avg, top5.avg, SCE, ECE) #auroc)


@torch.no_grad()
def test_temp(testloader, model, criterion, accelerator:Accelerator, args):

    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    all_targets = None
    all_outputs = None

    # switch to evaluate mode
    model.eval()

    bar = tqdm(enumerate(testloader), total=len(testloader), disable=(not accelerator.is_main_process))
    auroc_list = []
    for batch_idx, (inputs, targets) in bar:
        #auroc_metric = AUROC(task='multiclass', num_classes=100)
        # inputs, targets = inputs.cuda(), targets.cuda()
        #print(inputs)
        # compute output
        if "bert" in args.model or "bert" in args.teacher:
            inputs['input_ids'] = inputs['input_ids'].squeeze(1)
            outputs = model(inputs['input_ids'])
        else:
            outputs = model(inputs)
        
        if isinstance(outputs, SequenceClassifierOutput):
            outputs = outputs.logits
        
        loss = criterion(outputs, targets)
        
        prec1, prec3, prec5  = accuracy(outputs.data, targets.data, topk=(1, 3, 5))
        
        if args.dataset == "cifar100":
            targets_for_auc_calculation = F.one_hot(targets.data, num_classes=100)
        elif args.dataset == "cifar10":
            targets_for_auc_calculation = F.one_hot(targets.data, num_classes=10)
        elif args.dataset == "tinyimagenet":
            targets_for_auc_calculation = F.one_hot(targets.data, num_classes=200)
        elif args.dataset == "20_newsgroup":
            targets_for_auc_calculation = F.one_hot(targets.data, num_classes=20)
        elif args.dataset == "tiny_imagenet":
            targets_for_auc_calculation = F.one_hot(targets.data, num_classes=200)
 
        outputs_np = outputs.detach().cpu().numpy()
        targets_np = targets_for_auc_calculation.detach().cpu().numpy()

        auroc_batch_list = []
        for target, output in zip(targets_np, outputs_np):
            auroc = roc_auc_score(target, output, multi_class='ovr')
            auroc_batch_list.append(auroc)
        auroc_batch_mean = sum(auroc_batch_list) / len(auroc_batch_list)
        auroc_list.append(auroc_batch_mean)

        # gather metrics and outputs
        prec1, prec3, prec5, loss, auroc = accelerator.gather((prec1, prec3, prec5, loss, auroc))
        outputs, targets = accelerator.gather((outputs, targets))
        #print(outputs, targets)
        #print(targets.shape)
	    #auroc = auroc_metric(outputs, targets)
   
        if "bert" in args.model or "bert" in args.teacher:
            inputs_ids = inputs['input_ids']
        else:
            inputs_ids = inputs
        
        if accelerator.is_main_process:
            losses.update(loss.mean().item(), inputs_ids.size(0))
            top1.update(prec1.mean().item(), inputs_ids.size(0))
            top3.update(prec3.mean().item(), inputs_ids.size(0))
            top5.update(prec5.mean().item(), inputs_ids.size(0))

            targets = targets.cpu().numpy()
            outputs = outputs.cpu().numpy()

            if all_targets is None:
                all_outputs = outputs
                all_targets = targets
            else:
                all_targets = np.concatenate([all_targets, targets], axis=0)
                all_outputs = np.concatenate([all_outputs, outputs], axis=0)
    
        # plot progress
        bar.set_postfix_str('({batch}/{size}) Loss: {loss:.8f} | top1: {top1: .4f} | top3: {top3: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    top5=top5.avg,
                    ))
    
    if accelerator.is_main_process:
        ECE = ECELoss().loss(all_outputs, all_targets, n_bins=15)
        SCE = SCELoss().loss(all_outputs, all_targets, n_bins=15)
        auroc = sum(auroc_list) / len(auroc_list)
        AECE_class = AdaptiveECELoss(n_bins=15)
        all_outputs = torch.from_numpy(all_outputs)
        all_targets = torch.from_numpy(all_targets)
        AECE = AECE_class(all_outputs, all_targets)
    else:
        ECE = None
        SCE = None
        auroc = None
        AECE = None

    
    return (losses.avg, top1.avg, top3.avg, top5.avg, SCE, ECE, AECE, auroc) #auroc)

@torch.no_grad()
def get_logits_from_model_dataloader(testloader, model):
    """Returns torch tensor of logits and targets on cpu"""
    # switch to evaluate mode
    model.eval()

    all_targets = None
    all_outputs = None

    bar = tqdm(testloader, total=len(testloader), desc="Evaluating logits")
    for inputs, targets in bar:
        inputs = inputs.cuda()
        # compute output
        outputs = model(inputs)
        # to numpy
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()

        if all_targets is None:
            all_outputs = outputs
            all_targets = targets
        else:
            all_targets = np.concatenate([all_targets, targets], axis=0)
            all_outputs = np.concatenate([all_outputs, outputs], axis=0)

    return torch.from_numpy(all_outputs), torch.from_numpy(all_targets)

    
