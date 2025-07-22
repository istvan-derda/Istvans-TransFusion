#!/bin/bash
#SBATCH -J TransFusion
#SBATCH --time=03:00:00
#SBATCH --array=2
#SBATCH --partition=clara
#SBATCH --gpus=rtx2080ti
#SBATCH --mem=8G
#SBATCH -o jobfiles/%x_%A_%a.out
#SBATCH -e jobfiles/%x_%A_%a.err

module purge
pip freeze --user | xargs pip uninstall -y

module load PyTorch/1.12.1-foss-2021b-CUDA-11.5.2 
pip install pandas scikit-learn tqdm scipy einops tensorboard numba pyprojroot 

python train.py
