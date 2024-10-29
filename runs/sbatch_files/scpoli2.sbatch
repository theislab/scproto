#!/bin/bash
#SBATCH --job-name=scpoli
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --output=output2.txt
#SBATCH --error=error2.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate apex-env
# python main.py scpoli --experiment_name scpoli-108e --latent_dims 8 --fine_tuning_epochs 0 --epochs 300
# python main.py scpoli --experiment_name scpoli108e-le --latent_dims 8 --fine_tuning_epochs 0 --epochs 108 --linear_eval

# python main.py scpoli --latent_dims 8 --batch_size 1024 --experiment_name scpoli-pretrained --fine_tuning_epochs 0
# python main.py scpoli --latent_dims 8 --batch_size 1024 --experiment_name scpoli-both --fine_tuning_epochs 250 --pretraining_epochs 50
python main.py scpoli --latent_dims 8 --fine_tuning_epochs 150 --pretraining_epochs 150 --experiment_name scpoli --training_type semi_supervised --dataset_id hlca --batch_size 1024

# --dataset_id hlca
