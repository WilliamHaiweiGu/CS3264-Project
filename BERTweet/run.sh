#!/bin/sh
#SBATCH --job-name=BERTweet
#SBATCH --time=3:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:h100-96
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
hostname 
nvidia-smi
python train.py
echo -e "\nJob completed at $(date)"