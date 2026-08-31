# wss-rkhs
Source code for the paper "A Unified Variational Framework for Deep Weakly Supervised Image Segmentation"

## Preparing
Download the ResNet-101 weight file (.pth) and put it into the folder > model/pretrained
Create the folder > checkpoint

## Data Preparation
To be added...

## Training
We offer the training code for our method and also the comparison methods in the same framework. The codes are 

> train_rkhs.py (ours)
> train_rkhs_ms.py (Chan-Vese energy with pCE)
> train_rkhs_ncut.py (Normalized Graph Cut Term with pCE)
> train_rkhs_pce (pCE only)

respectively. The corresponding training setting can be modified in the YAML files in > configs
To start training, simply run one of the files above.

## Eval
Run
> eval.py
 
