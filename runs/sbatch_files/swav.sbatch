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
# python main_run.py --experiment_name all_loss --num_prototypes 300 --prot_decoding_loss_scaler 5 --cvae_loss_scaler 0.0001
# python main_swav.py --experiment_name all-loss-300e --num_prototypes 300 --prot_decoding_loss_scaler 5 --cvae_loss_scaler 0.0001 --freezable_prototypes
# python main.py swav --experiment_name all_loss --num_prototypes 300 --only_eval --linear_eval --use_early_stopping --fine_tuning_epochs 100 --model_name_version 2
# python main.py swav --experiment_name swav-larger-bs --num_prototypes 300 --linear_eval --use_early_stopping --fine_tuning_epochs 100
# python main.py swav --experiment_name swav --num_prototypes 300 --linear_eval --use_early_stopping --fine_tuning_epochs 100
# python main.py swav --experiment_name swav --num_prototypes 300 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024
# python main.py swav --experiment_name swav --num_prototypes 300 --latent_dims 32 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024
# python main.py swav --experiment_name swav --num_prototypes 300 --latent_dims 32 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type
python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0 --model_version 1