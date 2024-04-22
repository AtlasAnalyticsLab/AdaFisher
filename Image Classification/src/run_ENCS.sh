#!/bin/bash
#SBATCH --account=m_damie
#SBATCH --gpus=1
#SBATCH -patlas
#SBATCH -Aatlas
#SBATCH --mem=128GB
#SBATCH -o _%x%J.out
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=<m_damie@live.concordia.ca>

nvidia-smi -L
module unload cuda/default
module load python/3.11.6
# export NCCL_P2P_DISABLE=1 # for atlas Analytics when we use torchrun
################################ TRAINING BATCH FILE ###############################
######## AadFisher Training ########
# torchrun --standalone --nproc_per_node=gpu /home/m_damie/AdaFisher/src/train.py --config configs/adam.yaml --dist True
# python3 /home/m_damie/AdaFisher/src/train.py --config configs/AdaFisherCNN.yaml # you can change on the yaml file for AdaFisherW
######## Adam Training ########
# python3 /home/m_damie/AdaFisher/src/train.py --config configs/adamCNN.yaml  # You can change on the yaml file for AdamW

######## AdaHessian Training ########
python3 /home/m_damie/AdaFisher/src/train.py --config configs/AdaHessian.yaml

######## Shampoo Training ########
# python3 /home/m_damie/AdaFisher/src/train.py --config configs/ShampooCNN.yaml

######## Apollo Training ########
# python3 /home/m_damie/AdaFisher/src/train.py --config configs/Apollo.yaml

######## SAM Training ########
#python3 /home/m_damie/AdaFisher/src/train.py --config configs/SAM.yaml

######## kfac Training ########
# python3 /home/m_damie/AdaFisher/src/train.py --config configs/kfacCNN.yaml # you can change in the yaml file for ekfac
