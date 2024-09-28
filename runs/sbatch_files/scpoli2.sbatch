#!/bin/bash
#SBATCH --job-name=fatemeh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate apex-env
# python main.py scpoli --experiment_name scpoli-108e --latent_dims 8 --fine_tuning_epochs 0 --epochs 300
# python main.py scpoli --experiment_name scpoli108e-le --latent_dims 8 --fine_tuning_epochs 0 --epochs 108 --linear_eval

# python main.py scpoli --latent_dims 8 --batch_size 1024 --experiment_name scpoli-pretrained --fine_tuning_epochs 0
# python main.py scpoli --latent_dims 8 --batch_size 1024 --experiment_name scpoli-both --fine_tuning_epochs 250 --pretraining_epochs 50
python main.py scpoli --latent_dims 8 --batch_size 512 --fine_tuning_epochs 5 --pretraining_epochs 5 --experiment_name scpoli-semi-sample
