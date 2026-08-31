# wss-rkhs
Source code for the paper "A Unified Variational Framework for Deep Weakly Supervised Image Segmentation" (accepted by the Journal of Mathematical Imaging and Vision, to be published)

## Preparing
Download the ResNet-101 weight file (.pth) and put it into the folder 
> model/pretrained

Create the folder 
> checkpoints


## Data Preparation
To be added...

## Train
We provide training code for our method and the comparison methods in the same framework. The codes are 

> train_rkhs.py (ours)
> 
> train_rkhs_ms.py (Chan-Vese energy with pCE)
> 
> train_rkhs_ncut.py (Normalized Graph Cut Term with pCE)
> 
> train_rkhs_pce (pCE only)

respectively. The corresponding training setting can be modified in the YAML files in 
> configs

To start training, run one of the files above.

## Eval
Run
> eval.py
 
The generated images are stored in
> records/generated_images
