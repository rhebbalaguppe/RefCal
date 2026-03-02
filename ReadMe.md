ReadMeRefCal.md

# Source Code for the paper:

# Enhancing Deep Neural Network Reliability with Refinement and Calibration (ICLR 2026 Workshop on Trustworthy AI)

## We optimize both Refinement and Calibration to enhance reliability of DNN classifiers:

We intend to develop a strategy to train reliable DNN models that demonstrate high refinement, high accuracy, and good calibration (i.e., low calibration error), all at the same time. We describe our proposed framework, named RefCal. RefCal comprises two stages: (1) pretraining an encoder network to enforce refinement via contrastive training; (2) In the second stage we fine-tune a classifier head with the frozen encoder based on the non-contrastive losses for calibration and accuracy. The model thus trained achieves good refinement, calibration, and accuracy at the same time. We have experimented with various loss functions to achieve high accuracy and calibration in the second stage.

## Requirements:

We have tested our implementation of Refcal on the following environment:

-   Python 3.8.16 / Pytorch (>=1.8.0) / torchvision (>=0.9.0) / CUDA 11.3

## Training:

Please refer to the supplementary for hyperparameters.

### Training Stage 1 (Refinement):

To train a stage 1 model (say ResNet-50) on CIFAR-100 dataset , run the following command:

```
python main_refcal.py --batch_size 1024 \
  --learning_rate 0.5 \
  --dataset cifar100
  --temp 0.1 \
  --cosine

```

### Training Stage 2 (Calibration and Classification):

We illustrate three examples on how to incorporate calibration as shown below:

_Example 1: FL+MDCA_

To train a classifier using a pretrained Stage1 model (ResNet-50) using _FL+MDCA_ loss function to calibrate on CIFAR-100 dataset, run the following command:

```
python main_linear.py --batch_size 512 \
  --learning_rate 5 \
  --dataset cifar100
  --loss_function FL+MDCA \
  --gamma 2\ 
  --ckpt /path/to/stage1_model.pth

```

_Example 2: MBLS_

To train a classifier using a pretrained Stage1 model (ResNet-50) using _Cross Entropy+MBLS_ loss function to calibrate on CIFAR-100 dataset, run the following command:

```
python main_linear.py --batch_size 512 \
  --learning_rate 5 \
  --dataset cifar100
  --loss_function MBLS \
  --alpha 0.01\ 
  --ckpt /path/to/stage1_model.pth

```

_Example 3: LogitNorm_

To train a classifier using a pretrained Stage1 model (ResNet-50) using _LogitNorm_ loss function to calibrate on CIFAR-100 dataset , run the following command:

```
python main_linear.py --batch_size 512 \
  --learning_rate 5 \
  --dataset cifar100
  --loss_function logit_norm \
  --temp 0.01\ 
  --ckpt /path/to/stage1_model.pth

```

You can plugin any values you like for `--alpha` (smoothing weight α\\alphaα) and `--gamma` (focusing parameters γ\\gammaγ) arguments.
