from .resnet import resnet18
from .resnet import _get_cifar_resnext, _get_cifar_wrn
from .resnet_cifar import _get_cifar_resnet
from .convnet_cifar import _get_cifar_convnet
from .resnet_tinyimagenet import _get_tinyimagenet_resnet
from .resnet_imagenet import _get_imagenet_resnet
from .densenet import _get_densenet
from .mobilenetv2 import _get_mobilenetv2
from .shufflenetv2 import _get_shufflenetv2
from .shufflenetv1 import _get_shufflenetv1
from .densenet import _get_densenet
from .wrn import _get_wrn
#from .vit import _get_cifar_vit
#from vit_pytorch.distill import DistillableViT, DistillWrapper
from .cvt import _get_cvt
from .bert import BERTForTextClassification

def get_model(model, dataset, args):
    if "cifar" in dataset:
#        if model=="resnet18" and dataset=="cifar100":
#            return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)
#        elif model=="resnet18" and dataset=="cifar10":
#            return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)
        if model=="resnet18":
             if dataset=="cifar100":
                return resnet18(num_classes=100)
             elif dataset=="cifar10":
                return resnet18(num_classes=10)
        elif "resnet" in model:
            return _get_cifar_resnet(model, dataset)
        elif "convnet" in model:
            return _get_cifar_convnet(model, dataset)
        elif "mobilenetv2" in model:
            return _get_mobilenetv2(model, dataset)
        elif "shufflenetv1" in model:
            return _get_shufflenetv1(model, dataset)
        elif "shufflenetv2" in model:
            return _get_shufflenetv2(model, dataset)
        elif model == "wrn_50_2":
            return _get_cifar_wrn(model, dataset)
        elif "wrn" in model and model != "wrn_50_2":
            return _get_wrn(model, dataset)
        elif "resnext" in model:
            return _get_cifar_resnext(model, dataset)
        elif "densenet" in model:
            return _get_densenet(model, dataset)
        elif "cvt" in model:
            return _get_cvt(model, dataset)
        #elif "VanillaViT" in model:
        #    return _get_cifar_vit(model, dataset)
        #elif "vit" in model:
        #    return DistillableViT(image_size=224,
        #                          patch_size=16,
        #                          num_classes=100,
        #                          dim=1024,
        #                          depth=6, heads=6, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
    elif "tiny_imagenet" in dataset:
        if "resnet" in model:
            return _get_tinyimagenet_resnet(model, dataset)
        elif "mobilenetv2" in model:
            return _get_mobilenetv2(model, dataset)
        elif "shufflenetv1" in model:
            return _get_shufflenetv1(model, dataset)
        elif "shufflenetv2" in model:
            return _get_shufflenetv2(model, dataset)
        elif "densenet" in model:
            return _get_densenet(model, dataset)
        elif model == "wrn_50_2":
            return _get_cifar_wrn(model, dataset)
        elif "wrn" in model and model != "wrn_50_2":
            return _get_wrn(model, dataset)
    elif "imagenet" in dataset:
        if "resnet" in model:
            return _get_imagenet_resnet(model, dataset)
    elif "20_newsgroup" in dataset:
        if "bert-base-uncased" in model:
            return BERTForTextClassification(vocab_size=30522, num_classes=20, hidden=768, n_layers=12, attn_heads=12, dropout=0.1)
