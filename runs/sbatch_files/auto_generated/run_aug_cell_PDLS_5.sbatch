#!/bin/bash
#SBATCH --job-name=aug_cell_PDLS_5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --output=slurm-job-out/aug_cell_PDLS_5/output.txt
#SBATCH --error=slurm-job-out/aug_cell_PDLS_5/error.txt
#SBATCH --nice=10000
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_reservation
#SBATCH --gres=gpu:1
#SBATCH --reservation=rocky_linux_9_test

source ~/.bashrc
conda activate apex-env

python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 \
--augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 \
--prot_decoding_loss_scaler 5 --model_version 1 \
--dimensionality_reduction None --k_neighbors 10 --experiment_name aug_cell_PDLS_5
