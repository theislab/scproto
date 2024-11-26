#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --nice={nice_value}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --gres={gres}

source ~/.bashrc
conda activate {conda_env}

python main.py swav --num_prototypes {num_prototypes} --latent_dims {latent_dims} --batch_size {batch_size} \
--augmentation_type {augmentation_type} --epsilon {epsilon} --cvae_loss_scaler {cvae_loss_scaler} \
--prot_decoding_loss_scaler {prot_decoding_loss_scaler} --model_version {model_version} \
--dimensionality_reduction {dimensionality_reduction} --k_neighbors {k_neighbors} --experiment_name {experiment_name} \
--fine_tuning_epochs {fine_tuning_epochs} --pretraining_epochs {pretraining_epochs} --training_type {training_type} \
--dataset_id {dataset_id} --workers {workers} --job_name {job_name} --decodable_prototypes {decodable_prototypes} \
--cvae_epochs {cvae_epochs} --prot_init {prot_init} --checkpoint_freq {checkpoint_freq} --umap_checkpoint_freq {umap_checkpoint_freq} \
--sinkhorn_iterations {sinkhorn_iterations}
