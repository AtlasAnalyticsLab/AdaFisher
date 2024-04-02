#!/bin/bash 

################################ TRAINING BATCH FILE ###############################
######## AadFisher Training ########
python train.py --config configs/AdaFisherViT.yaml # you can change on the yaml file for AdaFisherW

######## Adam Training ########
#python train.py --config configs/adamCNN.yaml # You can change on the yaml file for AdamW

######## AdaHessian Training ########
#python train.py --config configs/AdaHessian.yaml

######## Shampoo Training ########
#python train.py --config configs/Shampoo.yaml

######## Apollo Training ########
#python train.py --config configs/Apollo.yaml

######## SAM Training ########
#python train.py --config configs/SAM.yaml

######## kfac Training ########
#python train.py --config configs/kfac.yaml # you can change in the yaml file for ekfac