#!/bin/bash
#SBATCH --account=m_damie
#SBATCH --gpus=1 -w virya1
#SBATCH --mem=128GB
#SBATCH -o _%x%J.out
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=<m_damie@live.concordia.ca>

nvidia-smi -L
module load anaconda/default
module load cuda/default