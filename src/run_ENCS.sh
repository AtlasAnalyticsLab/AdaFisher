#!/bin/bash
#SBATCH --account=m_damie
#SBATCH --gpus=1 -w virya4
#SBATCH --mem=128GB
#SBATCH -o _%x%J.out
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=<m_damie@live.concordia.ca>

nvidia-smi -L
module load anaconda/default
module load cuda/default
################################ TRAINING BATCH FILE ###############################
######## AadFisher Training ########
#python3 /home/m_damie/AdaFisher/src/train.py --config configs/AdaFisher.yaml # you can change on the yaml file for AdaFisherW

######## Adam Training ########
#python3 /home/m_damie/AdaFisher/src/train.py --config configs/adam.yaml # You can change on the yaml file for AdamW

######## AdaHessian Training ########
#python3 /home/m_damie/AdaFisher/src/train.py --config configs/AdaHessian.yaml

######## Shampoo Training ########
#python3 /home/m_damie/AdaFisher/src/train.py --config configs/Shampoo.yaml

######## Apollo Training ########
#python3 /home/m_damie/AdaFisher/src/train.py --config configs/Apollo.yaml

######## SAM Training ########
#python3 /home/m_damie/AdaFisher/src/train.py --config configs/SAM.yaml

######## kfac Training ########
python3 /home/m_damie/AdaFisher/src/train.py --config configs/kfac.yaml # you can change in the yaml file for ekfac