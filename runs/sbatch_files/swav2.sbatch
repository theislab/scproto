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

# python main.py swav --experiment_name swav-epsilon-0.02 --num_prototypes 300 --latent_dims 32 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --save_ref_umap

# python main.py swav --experiment_name swav-epsilon-0.03 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.03 --save_ref_umap
# python main.py swav --experiment_name swav-epsilon-0.03 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type knn --epsilon 0.03 --save_ref_umap


# python main.py swav --experiment_name swav-only-ep0.03 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type knn --epsilon 0.03 --save_ref_umap --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0
# python main.py swav --experiment_name swav-only-ep0.02 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type knn --epsilon 0.02 --save_ref_umap --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0
# python main.py swav --experiment_name swav-only-ep0.03 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.03 --save_ref_umap --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0
# python main.py swav --experiment_name swav-only-ep0.02 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --save_ref_umap --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0
# python main.py swav --experiment_name so-e2-e500 --epochs 500 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type knn --epsilon 0.02 --save_ref_umap --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0

# python main.py swav --experiment_name swav-epsilon-0.01 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.01 --save_ref_umap
# python main.py swav --experiment_name swav-epsilon-0.02 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type knn --epsilon 0.02 --save_ref_umap

# python main.py swav --experiment_name s-inter-e2 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --save_ref_umap --cvae_loss_scaler 0
# python main.py swav --experiment_name s-inter-e3 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.03 --save_ref_umap --cvae_loss_scaler 0

# python main.py swav --experiment_name swav15 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.015 --save_ref_umap

# python main.py swav --experiment_name pswav2 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --use_projector
# python main.py swav --experiment_name pswav2 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type knn --epsilon 0.02 --use_projector

# python main.py swav --experiment_name tuned-pswav2 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --use_projector --hidden_mlp 32 --swav_dim 8
# python main.py swav --experiment_name pswav3 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.03 --use_projector 

# python main.py swav --experiment_name pswav --num_prototypes 300 --latent_dims 16 --batch_size 1024 --augmentation_type cell_type --epsilon 0.05 --use_projector --linear_eval --use_early_stopping
# python main.py swav --experiment_name pswav15 --num_prototypes 300 --latent_dims 16 --batch_size 1024 --augmentation_type cell_type --epsilon 0.015 --use_projector --linear_eval --use_early_stopping
# python main.py swav --experiment_name pso2 --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --use_projector --linear_eval --use_early_stopping --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0

# python main.py swav --experiment_name pswav2 --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --use_projector --linear_eval --use_early_stopping


# python main.py swav --experiment_name so2-pparam --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0

# python main.py swav --experiment_name pso2-hidden64-swav16 --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0 --hidden_mlp 64 --swav_dim 16 --use_projector

# python main.py swav --experiment_name so2-rerun --num_prototypes 300 --latent_dims 8 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0 --model_version 1

# python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0 --hidden_mlp 64 --swav_dim 16 --use_projector
# python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --model_version 1 --epochs 500 --experiment_name swav_onlye500
# python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 5 --linear_eval --use_early_stopping --fine_tuning_epochs 100 --hidden_mlp 64 --swav_dim 16 --use_projector
# python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0 --epochs 1000 --experiment_name so-1000
# python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0 --experiment_name reproduce-try --model_version 1
# python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type cell_type --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0 --experiment_name r3 --model_version 1

# python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type scanpy_knn --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0 --model_version 1 --dimensionality_reduction pca --k_neighbors 50 --experiment_name seacell --workers 5
python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type community --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 0 --model_version 1 --dimensionality_reduction pca --k_neighbors 50 --experiment_name seacell

# python main.py swav --num_prototypes 300 --latent_dims 8 --batch_size 1024 --augmentation_type scanpy_knn --epsilon 0.02 --cvae_loss_scaler 0 --prot_decoding_loss_scaler 5 --model_version 1 --dimensionality_reduction pca --k_neighbors 50 --experiment_name seacell --workers 5
