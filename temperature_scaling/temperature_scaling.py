import os
import pandas as pd
import random

import csv

from torchmetrics import AUROC

import numpy as np

import transformers
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, DistilBertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from sys import argv

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path, Logger

from solvers.runners import test, test_temp, train_student
from solvers.kdloss import VanillaKD, UnsymVanillaKD

from calibration_library.calibrators import TemperatureScaling

from models import get_model
from datasets import dataloader_dict

import logging
from datetime import datetime

import accelerate

from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    write_csv = 1

    # input size does not change
    torch.backends.cudnn.benchmark = True # false means deterministic outputs, but less throughput

    # set up accelerator
    accelerator = accelerate.Accelerator()

    # parse arguments
    args = parse_args()

    # set seeds
    accelerate.utils.set_seed(args.seed)
    
    # prepare save path
    if args.current_time:
        current_time = args.current_time
    else:
        current_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    model_save_pth = f"{args.checkpoint}/{args.dataset}/{args.teacher}/{current_time}_TempScale={args.custom_temp}"
    checkpoint_dir_name = model_save_pth

    if not os.path.isdir(model_save_pth):
        mkdir_p(model_save_pth)

    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.FileHandler(filename=os.path.join(model_save_pth, "train_log.txt")),
                            logging.StreamHandler()
                        ])
    logging.info(f"Setting up logging folder : {model_save_pth}")
    logging.info(args)
    logging.info(argv)
    logging.info(f"GPUs used: {torch.cuda.device_count()}")

    accelerator.wait_for_everyone()

    logging.info(f"Using teacher model : {args.teacher}")
    logging.info(f"loading teacher model from: {args.teacher_path}")

    if args.pretrained_model == "BertForSequenceClassification":
        logging.info(f"Using model : {args.teacher}")
        logging.info(f"loading BertForSequenceClassification model from: {args.pretrained_model}")
        teacher = BertForSequenceClassification.from_pretrained(args.teacher, num_labels=20,
                                                              max_length=512)
    elif args.pretrained_model == "DistilBertForSequenceClassification":
        logging.info(f"Using model : {args.teacher}")
        logging.info(f"loading DistilBertForSequenceClassification model from: {args.pretrained_model}")
        teacher = DistilBertForSequenceClassification.from_pretrained(args.teacher, num_labels=20,
                                                              max_length=512)
    else:
        logging.info(f"Using model : {args.teacher}")
        logging.info(f"Applying Smoothing on the model with temperature={args.temp}")
        teacher = get_model(args.teacher, args.dataset, args)
   
    if args.use_parallel:
       teacher = torch.nn.DataParallel(teacher, device_ids=range(torch.cuda.device_count()))
       cudnn.benchmark = True

    # load teacher model
    saved_model_dict = torch.load(os.path.join(args.teacher_path, "model_best.pth"), map_location=accelerator.device)

    if "dataset" in saved_model_dict:
        assert saved_model_dict["dataset"] == args.dataset, \
            "Teacher not trained with same dataset as the student"
    if "net" in saved_model_dict:
        print(teacher.load_state_dict(saved_model_dict['net']))
    elif "state_dict" in saved_model_dict:
        print(teacher.load_state_dict(saved_model_dict['state_dict']))
    else:
        print(teacher.load_state_dict(saved_model_dict))

    # prepare model
    #accelerator.print(f"Using student model : {args.model}")
    #student = get_model(args.model, args.dataset, args)

    # set up dataset
    accelerator.print(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)

    #accelerator.print(f"Setting up optimizer : {args.optimizer}")

    #optimizer = optim.SGD(student.parameters(), 
    #                        lr=args.lr, 
    #                        momentum=args.momentum, 
    #                        weight_decay=args.weight_decay,
    #                        nesterov=True)
    
    # use vanilla KD with default params for now
    #criterion = VanillaKD(temp=args.T, distil_weight=args.Lambda)
    #criterion = UnsymVanillaKD(temp_t=args.T_t, temp_s=args.T_s, distil_weight=args.Lambda)
    # teacher does not need to be DDP to train student
    trainloader, valloader, testloader, teacher = accelerator.prepare(
        trainloader, valloader, testloader, teacher
    )
    
    test_criterion = torch.nn.CrossEntropyLoss()
   
    # Temperature scaling:
    if args.custom_temp is None:
        teacher = TemperatureScaling(base_model=teacher, T=args.temp)
        teacher.calibrate(valloader, args)
    else:
        teacher = TemperatureScaling(base_model=teacher, T=args.custom_temp)
    
    save_checkpoint({
	    'state_dict': teacher.state_dict(),
	    'dataset' : args.dataset,
	    'model' : args.teacher
	}, is_best=True, checkpoint=model_save_pth)

    #train_loss, top1_train, _, _, sce_score_train, ece_score_train, auroc_train = test_temp(trainloader, teacher, test_criterion, accelerator, args)
    test_loss, top1, top3, top5, sce_score, ece_score, aece, auroc_test = test_temp(testloader, teacher, test_criterion, accelerator, args)
    csv_filename = "TS_results.csv"
    if write_csv:
         with open(csv_filename, mode='a', newline='') as csv_file:
             writer = csv.writer(csv_file)
             #writer.writerow(["Temperature", "Top1", "ECE", "SCE")
             writer.writerow([args.dataset, args.teacher, args.teacher_path, args.custom_temp, top1, ece_score, sce_score, aece])
    print("Loss, path, Top1, ECE, SCE, AUROC, AECE")
    #print("TRAINING: ", train_loss, args.teacher_path, top1_train, ece_score_train, sce_score_train, auroc_test)
    print("TESTING: ", test_loss, args.teacher_path, top1, ece_score, sce_score, auroc_test, aece)
    
    
    # teacher does not need to be DDP to train student
    #trainloader, valloader, testloader, student, optimizer = accelerator.prepare(
    #    trainloader, valloader, testloader, student, optimizer
    #)
    
    # defining lr_scheduler here since number of steps can change depending upon GPUs for each process
    #if args.scheduler == "multistep":
    #    logging.info(f"Step sizes : {args.schedule_steps} | lr-decay-factor : {args.lr_decay_factor}")
    #    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_steps, gamma=args.lr_decay_factor)
    #elif args.scheduler == "warmupcosine":
    #    total_iters = int(len(trainloader) * args.epochs)
    #    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, args.warmup, total_iters)
    #elif args.scheduler == "cosine":
        #total_iters = int(args.epochs)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iters, verbose=True)

    #accelerator.print("Using the scheduler:", scheduler)
    
    #start_epoch = args.start_epoch
    
    #best_acc = 0.
    #best_sce = float("inf")
    #best_acc_stats = {"top1" : 0.0}

    # set up logger
    #if accelerator.is_main_process:
    #    logger = Logger(os.path.join(checkpoint_dir_name, "train_metrics.txt"))
    #    logger.set_names(["lr", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc", "SCE", "ECE"])

    #for epoch in range(start_epoch, args.epochs):

    #    accelerator.print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, get_lr(optimizer)))
        
    #    train_loss, top1_train = train_student(trainloader, student, teacher, optimizer, criterion, scheduler, accelerator, args)
    #val_loss, top1_val, _, _, sce_score_val, ece_score_val = test(valloader, teacher, test_criterion, accelerator)
    #test_loss, top1, top3, top5, sce_score, ece_score = test(testloader, teacher, test_criterion, accelerator)

    #    if args.scheduler == "multistep" or args.scheduler == "cosine":
    #        scheduler.step()
    #    accelerator.wait_for_everyone()

    #    if accelerator.is_main_process:
    #        accelerator.print("End of epoch {} stats: train_loss : {:.4f} | val_loss : {:.4f} | top1_train : {:.4f} | top1 : {:.4f} | SCE : {:.5f} | ECE : {:.5f}".format(
    #            epoch+1,
                #train_loss,
                #test_loss,
                #top1_train,
                #top1,
                #sce_score,
                #ece_score
            #))

            # save best accuracy model
            #is_best = top1_val > best_acc
            #best_acc = max(best_acc, top1_val)

            #save_checkpoint({
            #        'epoch': epoch + 1,
            #        'state_dict': accelerator.unwrap_model(student).state_dict(),
            #        'optimizer' : optimizer.state_dict(),
            #        'scheduler' : scheduler.state_dict(),
            #        'dataset' : args.dataset,
            #        'model' : args.model
            #    }, is_best, checkpoint=model_save_pth)
        
            # Update best stats
            #if is_best:
            #    best_acc_stats = {
            #        "top1" : top1,
            #        "top3" : top3,
            #        "top5" : top5,
            #        "SCE" : sce_score, #        "ECE" : ece_score
            #    }
            #logger.append([get_lr(optimizer), train_loss, top1_train, val_loss, top1_val, test_loss, top1, sce_score, ece_score])

    #if accelerator.is_main_process:
    #    logging.info("training completed...")
    #    logging.info("The stats for best accuracy model on test set are as below:")
    #    logging.info(best_acc_stats)
#
    #    logger.append(["best_accuracy", 0, 0, 0, 0, 0, best_acc_stats["top1"], best_acc_stats["SCE"], best_acc_stats["ECE"]])

        # log results to a common file
    #    df = {
    #        "dataset" : [args.dataset],
    #        "teacher" : [args.teacher],
    #        "teacher_path" : [args.teacher_path],
    #        "student" : [args.model],
    #        "T_t" : [args.T_t],
    #        "T_s" : [args.T_s],
    #        "Lambda" : [args.Lambda],
    #        "top1" : [best_acc_stats["top1"]],
    #        "ECE" : [best_acc_stats["ECE"]],
    #        "SCE" : [best_acc_stats["SCE"]],
    #        "folder_path" : [checkpoint_dir_name],
    #        "checkpoint_train_loss" : [train_loss],
    #        "checkpoint_train_top1" : [top1_train],
    #        "checkpoint_val_loss" : [val_loss],
    #        "checkpoint_val_top1" : [top1_val],
    #        "checkpoint_test_loss" : [test_loss],
    #        "checkpoint_test_top1" : [top1],
    #        "checkpoint_test_top3" : [top3],
    #        "checkpoint_test_top5" : [top5],
    #        "checkpoint_test_sce" : [sce_score],
    #        "checkpoint_test_ece" : [ece_score]
    #    }
        #df =  pd.DataFrame(df)
        #result_folder = "results_csv"
        #os.makedirs(result_folder, exist_ok=True)
        #save_path_file = os.path.join(result_folder, "student_metrics_unsym.csv")
        #df.to_csv(save_path_file, mode='a', index=False, header=(not os.path.exists(save_path_file)))
