# Test robustness of model using CIFAR100-c dataset
# References:
# 1. https://github.com/hendrycks/robustness/blob/master/ImageNet-C/test.py
# 2. https://github.com/psh150204/AugMix/blob/master/main.py
import os
import torch
import numpy as np
import pandas as pd
import argparse
import yaml

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from metrics import get_all_metrics

#import torchvision.models as models
import torch.utils.model_zoo as model_zoo

#from utils.get_all_models import get_model


#Step I: # *********Import model definition file here *******
# Example: loading resnet here
# from models import resnet
# from models.mobilenet import mobilenetv2

# set parser
# parser = argparse.ArgumentParser(description="Evaluates robustness of CNNs")
# parser.add_argument("-wp", "--weights_pth", type=str,
#                     default="pretrained/r50_bestmodel.pth",
#                     help="model weights file path")
# parser.add_argument("-dp", "--dataset_pth", type=str,
#                     default="dataset/CIFAR100-C",
#                     help="Robustness evaluation dataset folder path")

# args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # '0,1,2' for 3 gpus

        

def show_performance_cifar(model, dataloader, 
                            distortion_name=None, 
                            device='cuda',opt=None):

    # Calculate error
    pred_total = []
    label_total = []

    model.eval()  # Put model in eval mode

    err, correct, total = 0,0,0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(data)

            if opt.perform_temp_scaling:
                output /= opt.temp

            _, pred = torch.max(output.data, 1)
            correct += (pred==target).sum().item()
            total += target.size(0)

            pred_total.append(output.detach().cpu().numpy())
            label_total.append(target.detach().cpu().numpy())

    err = 1 - correct / total
    correct = correct / total
    #print(f"Total correct prediction (%): {correct*100}")

    if distortion_name is not None: # For robustness
        print(f"Distortion: {distortion_name}, Err: {err}")
        print(f"Distortion: {distortion_name}, Correct: {correct}")
        print(f"Total images in {distortion_name}: {total}")
    else:
        print("For the non distorted version the metrices are")
    
    pred_total = np.concatenate(pred_total)
    label_total = np.concatenate(label_total)
    name='val'
    results = get_all_metrics(name, pred_total, label_total,opt,logits = True)
    print(' * Acc@1 {top1:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f} '
            .format(top1=results[name+'_top1'], auc=results[name+'_auc'], ece=results[name+'_ece'], sce=results[name+'_sce']))

    return err, correct,results

def show_performance_cifar_supcon(val_loader, model, classifier, opt,distortion_name=None):
    """validation"""
    model.eval()
    classifier.eval()

    pred_total = []
    label_total = []

    err, correct, total = 0,0,0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            
            if opt.perform_temp_scaling:
                output /= opt.temp
            
            if opt.loss_function == "AUCM_loss":
                #output = torch.sigmoid(output)
                pass
            _, pred = torch.max(output.data, 1)
            correct += (pred==labels).sum().item()
            total += labels.size(0)

            pred_total.append(output.detach().cpu().numpy())
            label_total.append(labels.detach().cpu().numpy())

    err = 1 - correct / total
    correct = correct / total
    #print(f"Total correct prediction (%): {correct*100}")

    if distortion_name is not None: # For robustness
        print(f"Distortion: {distortion_name}, Err: {err}")
        print(f"Distortion: {distortion_name}, Correct: {correct}")
        print(f"Total images in {distortion_name}: {total}")
    else:
        print("For the non distorted version the metrices are")
    
    
    pred_total = np.concatenate(pred_total)
    label_total = np.concatenate(label_total)
    name='val'
    results = get_all_metrics(name, pred_total, label_total,opt,logits = True)
    print(' * Acc@1 {top1:.3f} AUC {auc:.3f} ECE {ece:.5f} SCE {sce:.5f} '
            .format(top1=results[name+'_top1'], auc=results[name+'_auc'], ece=results[name+'_ece'], sce=results[name+'_sce']))
    
    return err, correct, results

def save_results(final_output,opt):
    final_array = []
    for i in final_output.keys():
        if i == 'non_distorted':
            temp_array = [i]
            order_data = ['accuracy','auroc','ece','sce','ace','smece']
            for j in order_data:
                temp_array.append(final_output[i][j])
            for j in range(30):
                temp_array.append(-9999)

            final_array.append(temp_array)
        else:
            temp_array = [i]
            order = [1,2,3,4,5,'total_distorted']
            order_data = ['accuracy','auroc','ece','sce','ace','smece']
            for j in order:
                for k in order_data:
                    temp_array.append(final_output[i][j][k])
            final_array.append(temp_array)
    final_array_np = np.array(final_array)
    df = pd.DataFrame(final_array_np,columns=['corruption','accuracy_1','auroc_1','ece_1','sce_1','ace_1','smece_1','accuracy_2','auroc_2','ece_2','sce_2','ace_2','smece_2','accuracy_3','auroc_3','ece_3','sce_3','ace_3','smece_3','accuracy_4','auroc_4','ece_4','sce_4','ace_4','smece_4','accuracy_5','auroc_5','ece_5','sce_5','ace_5','smece_5','accuracy_t','auroc_t','ece_t','sce_t','ace_t','smece_t'])
    df.to_csv(os.path.join(opt.save_folder,opt.loss_function + ".csv"),index=False)

def cal_mCE(model, dataset_root, 
            dataset_transforms, 
            dataset_name,
            device='cuda',opt=None,classifier=None):
    
    # All the distortions: Total 15
    distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                    'defocus_blur', 'glass_blur', 'motion_blur',
                    'zoom_blur', 'snow', 'frost',
                    'brightness', 'contrast', 'elastic_transform',
                    'pixelate', 'jpeg_compression', 'speckle_noise',
                    'gaussian_blur', 'spatter', 'saturate']
    
    final_output = {}

    # Creating dummy object of datasets.CIFAR100 class and replacing later with
    # cifa100-c data and labels
    if dataset_name=="cifar100":
        test_data = datasets.CIFAR100("./datasets", 
                                        train=False, 
                                        transform=dataset_transforms,
                                        download=True)
    elif dataset_name=="cifar10":
        test_data = datasets.CIFAR10("./datasets", 
                                        train=False, 
                                        transform=dataset_transforms,
                                        download=True)
    else:
        raise NotImplementedError("Only for CIFAR100 and CIFAR10")
    
    # Standard dataset accuracy:
    standard_test_loader = torch.utils.data.DataLoader(test_data,
                                                        batch_size=32,
                                                        shuffle=False,
                                                        num_workers=8,
                                                        pin_memory=True)
    
    if classifier == None:
        err, correct,results = show_performance_cifar(model,
                                        standard_test_loader,
                                        device=device,opt=opt
                                        )
    else:
        err, correct,results = show_performance_cifar_supcon(standard_test_loader,model,
                                        classifier,opt=opt)
    
    temp_dict = {'accuracy':results['val_top1'],'auroc':results["val_auc"],'ece':results['val_ece'],'sce':results['val_sce'],'ace':results['val_ace'],'smece':results['val_smece']}
    final_output['non_distorted'] = temp_dict

    print(f"Standard Err (%): {err*100}")
    print(f"Standard Correct (%): {correct*100}")
    print("---------------------------------------------------------------------------------------------------------------")    

    print("Performing evaluation on the distorted versions")

    # Calculate errors: mCE
    errors = []
    corrects = []
    for distortion_name in distortions:
        final_output[distortion_name] = {}

        print("Performing evaluation on the distortion :",distortion_name)
        
        full_data_pth = os.path.join(dataset_root, f"{distortion_name}.npy")
        full_labels_pth = os.path.join(dataset_root, "labels.npy")

        all_corrupted_data = np.load(full_data_pth)
        all_corrupted_labels = torch.LongTensor(np.load(full_labels_pth))
        
        test_data.data = all_corrupted_data
        test_data.targets = all_corrupted_labels

        testloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=32,
                                                shuffle=False,
                                                num_workers=8,
                                                pin_memory=True)

        # error rate for a distortion
        if classifier == None:
            err, correct,results = show_performance_cifar(model,
                                            testloader,
                                            distortion_name,
                                            device=device,
                                            opt=opt
                                            )
        else:
            err, correct,results = show_performance_cifar_supcon(testloader,model,
                                            classifier,
                                            distortion_name=distortion_name,
                                            opt=opt
                                            )
        
        temp_dict = {'accuracy':results['val_top1'],'auroc':results["val_auc"],'ece':results['val_ece'],'sce':results['val_sce'],'ace':results['val_ace'],'smece':results['val_smece']}
        final_output[distortion_name]['total_distorted'] = temp_dict
        
        # Collect all distortion rates to calculate mCE later
        errors.append(err)
        corrects.append(correct)

        print('Distortion: {:15s} | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100*err))
        print("----------------------------------------------------------------------------------------------------------------")

        #print for every sevirity level
            
        

        for z in range(5):
            print("performing evaluation for distortion : {} at a sevirity of {}".format(distortion_name,z+1))
            imagedata_per_sevirity = all_corrupted_data[z*10000:(z+1)*10000]
            labeldata_per_sevirity = all_corrupted_labels[z*10000:(z+1)*10000]
            
            test_data.data = imagedata_per_sevirity
            test_data.targets = labeldata_per_sevirity

            testloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=32,
                                                shuffle=False,
                                                num_workers=8,
                                                pin_memory=True)

            # error rate for a distortion
            if classifier == None:
                err, correct,results = show_performance_cifar(model,
                                            testloader,
                                            distortion_name,
                                            device=device,
                                            opt=opt
                                            )
            else:
                err, correct,results = show_performance_cifar_supcon(testloader,model,
                                            classifier,
                                            distortion_name=distortion_name,
                                            opt=opt
                                            )
            
            temp_dict = {'accuracy':results['val_top1'],'auroc':results["val_auc"],'ece':results['val_ece'],'sce':results['val_sce'],'ace':results['val_ace'],'smece':results['val_smece']}
            final_output[distortion_name][z+1] = temp_dict
            print('Distortion: {:15s} | sevirity level {:.2f} | CE (unnormalized) (%): {:.2f}'.format(distortion_name,z+1,100*err))
            print("----------------------------------------------------------------------------------------------------------------")
    save_results(final_output,opt)
    # Calculate and print mCE
    print('mCE (unnormalized) (%): {:.2f}'.format(100 * np.mean(errors)))


def load_best_model(cfg, model):

    bestmodelpth = os.path.join(cfg['bestmodel']['path'], cfg['bestmodel']['name'])
    bestmodel = torch.load(bestmodelpth) # load .pth file
    model.load_state_dict(torch.load(bestmodel['model']))
    print("Best model loaded!")

    return model


def mCE_cifar100(dataset_root, model, device='cuda',opt=None,classifier=None):
    
    print("Calculating Errors on CIFAR100 and CIFAR100-C")
    dataset_name = "cifar100"
    cifar_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.50707516,  0.48654887,  0.44091784),
                                            (0.26733429,  0.25643846,  0.27615047))
                        ])
#    test_dataset = torchvision.datasets.CIFAR100(root='./dataset',
#                                                train=False,
#                                                download=True,
#                                                transform=cifar_transforms)
 
    cal_mCE(model, dataset_root, 
            dataset_transforms=cifar_transforms, 
            dataset_name=dataset_name,
            device=device,opt=opt,classifier=classifier) 


def mCE_cifar10(dataset_root, model, device='cuda',opt=None,classifier=None):


    # Calculate err
    print("Calculating Errors on CIFAR10 and CIFAR10-C")
    dataset_name = "cifar10"
    cifar_transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.49139968,  0.48215841,  0.44653091),
                                            (0.24703223,  0.24348513,  0.26158784))
                        ])
#    test_dataset = torchvision.datasets.CIFAR100(root='./dataset',
#                                                train=False,
#                                                download=True,
#                                                transform=cifar_transforms)
 
    cal_mCE(model, dataset_root, 
            dataset_transforms=cifar_transforms, 
            dataset_name=dataset_name,
            device=device,classifier=classifier) 



# if __name__== "__main__":

    
#     # Step II: # Create model : Assumes that model definition file is imported
#     #model = resnet.resnet50(100)
#     model = mobilenetv2.MobileNetV2Wrapper(num_class=10)
#     model = torch.nn.DataParallel(model).cuda() # modules.layername saved if dataparallel was used while saving the model. Therefore need to wrap again with DataParallel when loading weights
#     print("Model created!")

#     # Step III: Load model with weights
#     model.load_state_dict(torch.load(args.weights_pth)['model'])
#     print("Model loaded with weights!")

#     # Step IV:
#     # calculate mCE on cifar
#     print(f"Evaluating ...")
#     mCE_cifar10(args.dataset_pth, model, device='cuda')


