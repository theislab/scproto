def get_defaults():
    defaults = {
        "dataset_id": "pbmc-immune",
        "model_name_version": 3.5,
        "num_prototypes": 128,  # swav specific or 8
        "hidden_dim": 64,
        "latent_dims": 8,  # swav specific
        # "batch_size_version": 2,
        "batch_size": 512,
        "fine_tuning_epochs": 0,
        # "custom_cross_val": False,
        # "description": "",
        "experiment_name": "",  # swav specific
        "condition_key": "study",
        "cell_type_key": "cell_type",
        # "epochs": 300,
        "linear_eval": False,
        "only_eval": False,
        "use_early_stopping": False,
        "pretraining_epochs": 300,
        "training_type": 'pretrain',  # semi_supervised, transfer_learning
        'pretrain_dataset_id': 'hlca',
        'finetune_dataset_id': 'pbmc-immune',
        
        "dump_name_version": 4,  # swav specific
        "nmb_crops": [8],  # swav specific
        "augmentation_type": "knn",  # swav specific
        "size_crops": [224],  # swav specific
        "min_scale_crops": [0.14],  # swav specific
        "max_scale_crops": [1],  # swav specific
        "crops_for_assign": [0, 1],  # swav specific
        "temperature": 0.1,  # swav specific
        "epsilon": 0.05,  # swav specific
        "sinkhorn_iterations": 3,  # swav specific
        "feat_dim": 8,  # swav specific
        "queue_length": 0,  # swav specific
        "epoch_queue_starts": 15,  # swav specific
        "base_lr": 4.8,  # swav specific
        "final_lr": 0,  # swav specific
        "freeze_prototypes_niters": 313,  # swav specific
        "wd": 1e-6,  # swav specific
        "warmup_epochs": 10,  # swav specific
        "start_warmup": 0,  # swav specific
        "cvae_reg": 0,  # swav specific
        "dist_url": "env://",  # swav specific
        "world_size": -1,  # swav specific
        "rank": 0,  # swav specific
        "local_rank": 0,  # swav specific
        "workers": 10,  # swav specific
        "checkpoint_freq": 25,  # swav specific
        "use_fp16": False,  # swav specific
        "sync_bn": "pytorch",  # swav specific
        "syncbn_process_group_size": 8,  # swav specific
        "seed": 31,  # swav specific
        "model": "",  # swav specific
        "optimizer": "",  # swav specific
        "lr_schedule": "",  # swav specific
        "queue": None,  # swav specific
        "train_loader": "",  # swav specific
        "training_stats": "",  # swav specific
        "device": "cuda",  # swav specific
        "freezable_prototypes": False,  # swav specific (should be true)
        "cvae_loss_scaler": 0.0001,  # swav specific
        "prot_decoding_loss_scaler": 5,  # swav specific
        "hidden_mlp": 1024,  # swav specific
        "swav_dim": 64,  # swav specific
        "use_projector": False,  # swav specific
        "model_version": 2,  # swav specific
        "train_decoder": False,  # swav specific
        "longest_path": 3,  # swav specific
        "dimensionality_reduction": None,  # swav specific
        'k_neighbors': 10,  # swav specific
        'model_type': 'swav',
        'job_name': '',
        'no_data': 'False',
        # 'is_swav': 0
    }
    return defaults
