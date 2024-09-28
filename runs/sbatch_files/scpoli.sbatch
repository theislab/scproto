#!/bin/bash
#SBATCH --job-name=fatemeh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate apex-env
# python main.py scpoli --experiment_name scpoli-108e --latent_dims 8 --fine_tuning_epochs 0 --epochs 300
python main.py scpoli --experiment_name scpoli300e --latent_dims 8 --fine_tuning_epochs 100 --linear_eval --use_early_stopping --batch_size 1024
# python main.py scpoli --experiment_name scpoli500e --latent_dims 8 --epochs 500 --fine_tuning_epochs 100 --linear_eval --use_early_stopping --batch_size 1024