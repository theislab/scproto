ABBREVIATIONS = {
    # SLURM-related parameters
    # "job_name": "job",
    "nodes": "N",
    "cpus_per_task": "CPUs",
    "memory": "mem",
    "output_file": "out",
    "error_file": "err",
    "nice_value": "nice",
    "partition": "part",
    "qos": "qos",
    "gres": "gres",

    # Conda environment
    "conda_env": "env",

    # Experiment parameters
    "augmentation_type": "aug",
    "dimensionality_reduction": "DR",
    "batch_size": "BS",
    "latent_dims": "LD",
    "num_prototypes": "NP",
    "epsilon": "eps",
    "cvae_loss_scaler": "cvaeLS",
    "prot_decoding_loss_scaler": "PDLS",
    "model_version": "MV",
    "k_neighbors": "kN",
    "experiment_name": "exp",
    'semi_supervised': 'semi',

    # Add any other parameters you need to abbreviate here
    "learning_rate": "LR",
    "optimizer": "opt",
    "dropout_rate": "dropout",
    "activation_function": "act",
    "epochs": "ep",
    "dataset_id": "ds",
    "shuffle": "shuf",
    "validation_split": "val_split",
    "batch_size": 'bs',
    "temperature": "temp",
    'epsilon': "eps",
    'freeze_prototypes_niters': 'frz',
    'prot_init': 'prtInit',
    'propagation_reg': 'propReg',
    'prot_emb_sim_reg': 'PEmbSimReg',
    'loss_type': 'loss',
    'training_type': 'train',
    # 'pretraining_epochs': 'pre-e',
    # 'fine_tuning_epochs': 'tune-e'
    'pretraining_epochs': 'pretrain',
    'fine_tuning_epochs': 'finetune'
}