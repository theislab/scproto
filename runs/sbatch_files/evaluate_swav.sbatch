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
# python main_run.py --experiment_name swav_loss_only --num_prototypes 300
# python evaluate_ssl.py --experiment_name all_loss --num_prototypes 300 --model_name_version 2 --use_fp16 False
python swav_test.py --experiment_name all_loss --num_prototypes 300 --model_name_version 2 --seed 2