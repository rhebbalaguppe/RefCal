from __future__ import print_function

import os
import sys
import argparse
import time
import math
import random


import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter,mixup_criterion,mixup_data
from util import adjust_learning_rate, warmup_learning_rate, accuracy,save_output_labels_and_conf
from util import set_optimizer, save_model
from networks.resnet_big import SupCEResNet
from custom_dataset import Custom_Dataset,Custom_Dataset_Imagenet,Custom_Dataset_stl10_binary,load_imagenet_lt_non_supcon

from metrics import get_all_metrics
from solvers.loss import loss_dict
from deterministic import seed_everything
from crl_utils import History
from imbalance_cifar import IMBALANCECIFAR100, IMBALANCECIFAR10,IMBALANCETINYIMAGENET
from torch.autograd import Variable
from robustness_standalone import mCE_cifar100,mCE_cifar10
from libauc.sampler import DualSampler
from libauc.datasets import CAT_VS_DOG
from ood_code.cal import test as cal_test
import matplotlib.pyplot as plt

import tensorboard_logger as tb_logger

import numpy as np

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100','cifar100_imb','cifar10_imb','imagenet','imagenet_lt','tiny_imagenet','tiny_imagenet_imb','stl10_binary','c2'], help='dataset')
    parser.add_argument('--imagenet_path',type=str,default='')
    parser.add_argument('--imbalance_ratio', default=0.1, type=float,
                        help='imbalance ratio of datasets')
    parser.add_argument('--eval_only', action='store_true',
                        help='perform only evaluation')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--corrupt', action='store_true',
                        help='perform only evaluation on corrupted dataset')
    parser.add_argument('--corrupted_dataset_path', type=str, default='',
                        help='path to corrupted dataset')
    parser.add_argument('--save_correct_labels', action='store_true',
                        help='saves the correct labels abd its corresponding confidence.')
    parser.add_argument('--save_images', action='store_true',
                        help='save output images')

    # other setting
    parser.add_argument('--loss_function', type=str, default='cross_entropy',
                        choices=["focal_loss","cross_entropy","LS","NLL+MDCA","LS+MDCA","FL+MDCA","brier_loss","NLL+DCA","MMCE","FLSD",'AUCM_loss','MBLS','AdaFocal','CRL','ACLS_loss','logit_norm'], help='type of loss that has to be applied to the linear classifier')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='1',
                        help='id for recording multiple runs')
    parser.add_argument('--delta', default=0.25, type=float,
                        help='delta to use in Huber Loss in MDCA')
    parser.add_argument('--alpha', default=5.0, type=float,
                        metavar='ALPHA', help='alpha to train Label Smoothing/MBLS with')
    parser.add_argument('--beta', default=10, type=float,
                        metavar='BETA', help='beta to train DCA/MDCA with')
    parser.add_argument('--gamma', default=1, type=float,
                        metavar='GAMMA', help='gamma to train Focal Loss with')
    
    #parametrs for adafocal
    parser.add_argument("--num-bins", type=int, default=15, dest="num_bins", help="Number of calibration bins")
    parser.add_argument("--adafocal-lambda", type=float, default=1.0, dest="adafocal_lambda", help="lambda for adafocal.")
    parser.add_argument("--adafocal-gamma-initial", type=float, default=1.0, dest="adafocal_gamma_initial", help="Initial gamma for each bin.")
    parser.add_argument("--adafocal-gamma-max", type=float, default=20.0, dest="adafocal_gamma_max", help="Maximum cutoff value for gamma.")
    parser.add_argument("--adafocal-gamma-min", type=float, default=-2.0, dest="adafocal_gamma_min", help="Minimum cutoff value for gamma.")
    parser.add_argument("--adafocal-switch-pt", type=float, default=0.2, dest="adafocal_switch_pt", help="Gamma at which to switch to inverse-focal loss.")
    parser.add_argument("--update-gamma-every", type=int, default=-1, dest="update_gamma_every", help="Update gamma every nth batch. If -1, update after epoch end.")
    
    #parameters for CRL
    parser.add_argument('--rank_weight', default=1, type=float,
                        metavar='rank_weight', help='rank weight to train CRL Loss with')
    parser.add_argument('--rank_target', default='softmax', type=str,
                        metavar='rank_target', help='rank target to train CRL Loss with')
    
    parser.add_argument('--seed', type=int, default=1234,
                        help='seed for the program')
    
    #parameters for mixup
    parser.add_argument('--mixup', action='store_true',
                        help='to use mixup')
    parser.add_argument('--mixup_alpha', default=1.0, type=float,
                        metavar='mixup_alpha', help='alpha for the mixup operation')
    
    #parameters for logitnorm and tempeature scaling.
    parser.add_argument('--temp', default=1.0, type=float,
                        metavar='temp', help='temp for logitnorm/temperature scaling')
    
    parser.add_argument('--perform_temp_scaling', action='store_true',
                        help='perform temp scaling')
    parser.add_argument('--caliberate', action='store_true',
                        help='to perform temperature scaling')
    
    #testing for ood 
    parser.add_argument('--test_ood', action='store_true',
                        help='testing performance on the ood svhn dataset')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_loss_function_{}_lr_{}_decay_{}_bsz_{}_trial_{}_seed_{}'.\
        format(opt.dataset, opt.model, opt.loss_function, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial,opt.seed)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    
    if opt.mixup:
        opt.model_name = '{}_mixup'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10' or opt.dataset == 'cifar10_imb':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100' or opt.dataset == 'cifar100_imb':
        opt.n_cls = 100
    elif opt.dataset == 'imagenet' or opt.dataset == 'imagenet_lt':
        opt.n_cls = 1000
    elif opt.dataset == 'tiny_imagenet' or opt.dataset == 'tiny_imagenet_imb':
        opt.n_cls = 200
    elif opt.dataset == 'stl10_binary' or opt.dataset == 'c2':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10' or opt.dataset == 'cifar10_imb':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        crop_size = 32
    elif opt.dataset == 'cifar100' or opt.dataset == 'cifar100_imb':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        crop_size = 32
    elif opt.dataset == 'imagenet' or opt.dataset == 'imagenet_lt' or opt.dataset == 'tiny_imagenet' or opt.dataset == 'tiny_imagenet_imb':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_size = 64
    elif opt.dataset == 'stl10_binary':
        mean = (0.4467, 0.4398, 0.4066)
        std = (0.2603, 0.2564, 0.2762)
        crop_size = 64
    elif opt.dataset == 'c2':
        mean = (0.33554432, 0.33554432, 0.33554432)
        std = (0.28430098, 0.2612929,  0.24912025)
        crop_size = 50
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.dataset != 'imagenet_lt':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'cifar10_imb':
        train_dataset = IMBALANCECIFAR10(root= opt.data_folder, imb_factor=opt.imbalance_ratio, rand_number=opt.seed, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=opt.data_folder, train=False, download=True, transform=val_transform)
    elif opt.dataset == 'cifar100_imb':
        train_dataset = IMBALANCECIFAR100(root= opt.data_folder, imb_factor=opt.imbalance_ratio, rand_number=opt.seed, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100(root=opt.data_folder, train=False, download=True, transform=val_transform)
    elif opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=os.path.join(opt.imagenet_path,'ILSVRC/Data/CLS-LOC/train'),transform=train_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(opt.imagenet_path,'ILSVRC/Data/CLS-LOC/val'),transform=val_transform)
    elif opt.dataset == 'imagenet_lt':
        train_data = load_imagenet_lt_non_supcon(data_root=opt.data_folder,batch_size=opt.batch_size,dataset=opt.dataset,phase='train',transform=train_transform)
        val_data = load_imagenet_lt_non_supcon(data_root=opt.data_folder,batch_size=opt.batch_size,dataset=opt.dataset,phase='test',transform=train_transform)   
    elif opt.dataset == 'tiny_imagenet':
        train_dataset = datasets.ImageFolder(root=os.path.join(opt.imagenet_path,'tiny-imagenet-200/train'),transform=train_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(opt.imagenet_path,'tiny-imagenet-200/val/images'),transform=val_transform)
    elif opt.dataset == 'tiny_imagenet_imb':
        train_dataset = IMBALANCETINYIMAGENET(root= os.path.join(opt.imagenet_path,'tiny-imagenet-200/train'), imb_factor=opt.imbalance_ratio, rand_number=opt.seed, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=os.path.join(opt.imagenet_path,'tiny-imagenet-200/val/images'),transform=val_transform)
    elif opt.dataset == 'stl10_binary':
        train_dataset = datasets.STL10(root=opt.data_folder,
                                       split='train',
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.STL10(root=opt.data_folder,
                                     split='test',
                                    transform=val_transform,
                                    download=True)
    elif opt.dataset == 'c2':
        train_data, train_targets = CAT_VS_DOG(root=opt.data_folder)
        val_data, val_targets = CAT_VS_DOG(root=opt.data_folder, train=False)
    else:
        raise ValueError(opt.dataset)
    
    if opt.dataset == 'imagenet' or opt.dataset == 'tiny_imagenet' or opt.dataset == 'tiny_imagenet_imb':
        train_data = Custom_Dataset_Imagenet(train_dataset.imgs, train_transform)
        val_data = Custom_Dataset_Imagenet(val_dataset.imgs, val_transform)
    elif opt.dataset == "stl10_binary":
        train_data = Custom_Dataset_stl10_binary(train_dataset.data,train_dataset.labels,'stl10_binary', train_transform)
        val_data = Custom_Dataset_stl10_binary(val_dataset.data,val_dataset.labels,'stl10_binary', val_transform)
    elif opt.dataset == "c2":
        train_data = Custom_Dataset(train_data, train_targets,'c2', train_transform)
        val_data = Custom_Dataset(val_data, val_targets,'c2', val_transform)
    elif opt.dataset == 'imagenet_lt':
        print("already created the dataset")
    else:
        train_data = Custom_Dataset(train_dataset.data,train_dataset.targets,'cifar', train_transform)
        val_data = Custom_Dataset(val_dataset.data,val_dataset.targets,'cifar', val_transform)
    
    if opt.loss_function == "AUCM_loss":
        train_sampler = DualSampler(train_data,opt.batch_size,sampling_rate=0.2)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = loss_dict[opt.loss_function](gamma=opt.gamma, alpha=opt.alpha, beta=opt.beta, loss=opt.loss_function, delta=opt.delta,temp=opt.temp,args=opt)
    test_criterion = loss_dict["cross_entropy"]()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        test_criterion = test_criterion.cuda()
        cudnn.benchmark = True

    return model, criterion, test_criterion


def train(train_loader, model, criterion, optimizer, epoch, opt,history=None):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    pred_total = []
    label_total = []

    end = time.time()
    for idx, (images, labels,iter_idx) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.dataset == 'c2':
            images = images.type(torch.FloatTensor)
            labels = labels.to(torch.int64)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        if opt.mixup:
            #implement the code for mixup
            images, labels_a, labels_b, lam = mixup_data(images,labels,opt.mixup_alpha)
            images, labels_a, labels_b = map(Variable, (images,labels_a, labels_b))

        # compute loss
        output = model(images)
        
        if not opt.mixup:
            if opt.loss_function == "CRL":
                loss = criterion(output,labels,args=opt,idx=iter_idx,history=history)
            else:
                loss = criterion(output, labels)
        else:
            if opt.loss_function == "CRL":
                loss = mixup_criterion(criterion,output,labels_a,labels_b,loss_function=opt.loss_function,lam=lam,args=opt,idx=iter_idx,history=history)
            else:
                loss = mixup_criterion(criterion,output,labels_a,labels_b,loss_function=opt.loss_function,lam=lam)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 1))
        top1.update(acc1[0], bsz)

        pred_total.append(output.detach().cpu().numpy())
        label_total.append(labels.detach().cpu().numpy())

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    pred_total = np.concatenate(pred_total)
    label_total = np.concatenate(label_total)
    name='train'
    #results = get_all_metrics(name, pred_total, label_total,opt,logits = True)
    #print(' * Acc@1 {top1:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f} '
    #        .format(top1=results[name+'_top1'], auc=results[name+'_auc'], ece=results[name+'_ece'], sce=results[name+'_sce']))
    
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    pred_total = []
    label_total = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels,iter_idx) in enumerate(val_loader):

            if opt.dataset == 'c2':
                images = images.type(torch.FloatTensor)
                labels = labels.type(torch.int64)
            
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            
            if opt.perform_temp_scaling:
                output /= opt.temp
            
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 1))
            top1.update(acc1[0], bsz)

            if opt.save_images:
                for img in images:
                    # mean = (0.5071, 0.4867, 0.4408)
                    # std = (0.2675, 0.2565, 0.2761)
                    unnormalize = transforms.Normalize(mean=[-0.5071 / 0.2675, -0.4867 / 0.2565, -0.4408 / 0.2761],std=[1 / 0.2675, 1 / 0.2565, 1 / 0.2761])
                    img = unnormalize(img)
                    img_np = img.cpu().detach().numpy()
                    img_np = np.transpose(img_np,(1,2,0))
                    #plt.imsave("<path>/test.jpg",img_np)

            pred_total.append(output.detach().cpu().numpy())
            label_total.append(labels.detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    pred_total = np.concatenate(pred_total)
    label_total = np.concatenate(label_total)
    name='val'
    results={}
    results = get_all_metrics(name, pred_total, label_total,opt,logits = True)
    print(' * Acc@1 {top1:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f} '
           .format(top1=results[name+'_top1'], auc=results[name+'_auc'], ece=results[name+'_ece'], sce=results[name+'_sce']))

    if opt.save_correct_labels:
        save_output_labels_and_conf(pred_total, label_total,opt)

    return losses.avg, top1.avg , results

def calibrate(val_loader, model, criterion, opt):
        
        min_error = float('inf')
        min_T = 1.0
        temp_list = []

        t = 0.2
        init_t = 0.2
        while init_t <= 10:
            temp_list.append(init_t)
            init_t += t

        for T in temp_list:
            error = 0.
            with torch.no_grad():

                for idx, (images, labels,iter_idx) in enumerate(val_loader):
                    images = images.float().cuda()
                    labels = labels.cuda()
                    bsz = labels.shape[0]

                    # forward
                    output = model(images)
                    output /= T
                    cur_error = criterion(output, labels)

                    error += cur_error.item()
            
            if error < min_error:
                min_T = T
                min_error = error

        print(f"Best Temperature for Calibration: {min_T}")
        return min_T

        

def main():
    best_acc = 0
    best_auroc = 0
    opt = parse_option()

    #applying seed 
    #seed_everything(opt.seed)
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(opt.seed)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion, test_criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model,criterion)

    if opt.loss_function == "CRL":
        correctness_history = History(len(train_loader.dataset))
    else:
        correctness_history = None

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if opt.eval_only:
        #write the code for evaluation

        #load the checkpoint
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)

        if opt.corrupt:
            if opt.dataset == 'cifar100':
                mCE_cifar100(opt.corrupted_dataset_path, model, device='cuda',opt=opt)
            elif opt.dataset == 'cifar10':
                mCE_cifar10(opt.corrupted_dataset_path, model, device='cuda',opt=opt)
            return

        loss, val_acc, results = validate(val_loader, model, test_criterion, opt)
        print(results)

        return
    
    if opt.caliberate:
        #write the code for caliberation

        #load the checkpoint
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)
        #min_T = calibrate(val_loader, model, criterion, opt)
        
        opt.temp = 2
        opt.perform_temp_scaling = True
        loss, val_acc, results = validate(val_loader, model, test_criterion, opt)
        print(results)
        return

    if opt.test_ood:
        #write the code for testing the ood code
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)
        loss, val_acc, results = validate(val_loader, model, test_criterion, opt)
        
        cal_test(nnName='resnet50',dataName='svhn',CUDA_DEVICE='cuda',epsilon=0.0014,temperature=1000,net1=model,test_criterion=test_criterion,opt=opt)
        return

    print("The hyper parametrs are")
    print(opt)
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt,history=correctness_history)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        #tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc, results = validate(val_loader, model, test_criterion, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            best_results_acc = results
            print("best model in terms of accuracy")
            print(best_results_acc)
            save_file = os.path.join(opt.save_folder, 'best_epoch_acc.pth')
            save_model(model, optimizer, opt, epoch, save_file)

        # if results['val_auc'] > best_auroc:
        #     best_acc_auroc = val_acc
        #     best_auroc = results['val_auc']
        #     best_results_auroc = results
        #     print("best model in terms of auroc")
        #     print(best_results_auroc)
        #     save_file = os.path.join(opt.save_folder, 'best_epoch_auroc.pth')
        #     save_model(model, optimizer, opt, epoch, save_file)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best model in terms of accuracy is having an accuracy of : {:.2f}'.format(best_acc))
    print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}'.format(
        best_results_acc['val_auc'], best_results_acc['val_ece'], best_results_acc['val_sce']))
    print(best_results_acc)

    # print('best model in terms of auroc is having an of accuracy: {:.2f}'.format(best_acc_auroc))
    # print('best AUC: {:.10f}\t best ECE: {:.10f}\t best SCE: {:.10f}'.format(
    #     best_results_auroc['val_auc'], best_results_auroc['val_ece'], best_results_auroc['val_sce']))
    # print(best_results_auroc)


if __name__ == '__main__':
    main()
