from interpretable_ssl.configs.defaults import *

def set_parser_defaults(parser, defaults):
    parser_keys = [action.dest for action in parser._actions]
    filtered_defaults = {key: defaults[key] for key in parser_keys if key in defaults}
    parser.set_defaults(**filtered_defaults)
    return parser

def add_trainer_parser_args(parser):
    defaults = get_trainer_defaults()

    # Define command line arguments using defaults
    parser.add_argument("--partially_train_ratio", type=float, default=defaults['partially_train_ratio'])
    parser.add_argument("--self_supervised", type=bool, default=defaults['self_supervised'])
    parser.add_argument("--dataset", default=defaults['dataset'])
    parser.add_argument("--num_prototypes", type=int, default=defaults['num_prototypes'])
    parser.add_argument("--hidden_dim", type=int, default=defaults['hidden_dim'])
    parser.add_argument("--latent_dims", type=int, default=defaults['latent_dims'])
    parser.add_argument("--batch_size_version", type=int, default=defaults['batch_size_version'])
    parser.add_argument("--batch_size", type=int, default=defaults['batch_size'])  # Allow batch size to be None
    parser.add_argument("--fine_tuning_epochs", type=int, default=defaults['fine_tuning_epochs'])
    parser.add_argument("--model_name_version", type=int, default=defaults['model_name_version'])

    return parser

def add_swav_parser_args(parser):
    # parser = add_scpoli_parser_args(parser)
    defaults = get_swav_defaults()
    # parser = set_parser_defaults(parser, defaults)
    
    parser.add_argument("--experiment_name", type=str, default=defaults['experiment_name'])

    # Define command line arguments using defaults
    parser.add_argument('--dataset', type=str, default=defaults['dataset'], help='Dataset to use')
    parser.add_argument('--dump_name_version', type=int, default=defaults['dump_name_version'], help='Dump name version')
    parser.add_argument('--nmb_crops', type=int, nargs='+', default=defaults['nmb_crops'], help='Number of crops')
    parser.add_argument('--augmentation_type', type=str, default=defaults['augmentation_type'], help='Type of augmentation')
    parser.add_argument('--size_crops', type=int, nargs='+', default=defaults['size_crops'], help='Size of crops')
    parser.add_argument('--min_scale_crops', type=float, nargs='+', default=defaults['min_scale_crops'], help='Minimum scale of crops')
    parser.add_argument('--max_scale_crops', type=float, nargs='+', default=defaults['max_scale_crops'], help='Maximum scale of crops')
    parser.add_argument('--crops_for_assign', type=int, nargs='+', default=defaults['crops_for_assign'], help='Crops for assignment')
    parser.add_argument('--temperature', type=float, default=defaults['temperature'], help='Temperature parameter')
    parser.add_argument('--epsilon', type=float, default=defaults['epsilon'], help='Epsilon parameter')
    parser.add_argument('--sinkhorn_iterations', type=int, default=defaults['sinkhorn_iterations'], help='Number of Sinkhorn iterations')
    parser.add_argument('--latent_dims', type=int, default=defaults['latent_dims'], help='Latent dimensions')
    parser.add_argument('--feat_dim', type=int, default=defaults['feat_dim'], help='Feature dimensions')
    parser.add_argument('--num_prototypes', type=int, default=defaults['num_prototypes'], help='Number of prototypes')
    parser.add_argument('--queue_length', type=int, default=defaults['queue_length'], help='Queue length')
    parser.add_argument('--epoch_queue_starts', type=int, default=defaults['epoch_queue_starts'], help='Epoch when queue starts')
    parser.add_argument('--epochs', type=int, default=defaults['epochs'], help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'], help='Batch size')
    parser.add_argument('--base_lr', type=float, default=defaults['base_lr'], help='Base learning rate')
    parser.add_argument('--final_lr', type=float, default=defaults['final_lr'], help='Final learning rate')
    parser.add_argument('--freeze_prototypes_niters', type=int, default=defaults['freeze_prototypes_niters'], help='Freeze prototypes iterations')
    parser.add_argument('--wd', type=float, default=defaults['wd'], help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=defaults['warmup_epochs'], help='Number of warmup epochs')
    parser.add_argument('--start_warmup', type=float, default=defaults['start_warmup'], help='Start warmup learning rate')
    parser.add_argument('--cvae_reg', type=float, default=defaults['cvae_reg'], help='CVAE regularization')
    parser.add_argument('--dist_url', type=str, default=defaults['dist_url'], help='URL used to set up distributed training')
    parser.add_argument('--world_size', type=int, default=defaults['world_size'], help='Number of processes')
    parser.add_argument('--rank', type=int, default=defaults['rank'], help='Rank of the process')
    parser.add_argument('--local_rank', type=int, default=defaults['local_rank'], help='Local rank of the process')
    parser.add_argument('--workers', type=int, default=defaults['workers'], help='Number of data loading workers')
    parser.add_argument('--checkpoint_freq', type=int, default=defaults['checkpoint_freq'], help='Checkpoint frequency')
    parser.add_argument('--use_fp16', type=bool, default=defaults['use_fp16'], help='Use FP16')
    parser.add_argument('--sync_bn', type=str, default=defaults['sync_bn'], help='Sync batch normalization type')
    parser.add_argument('--syncbn_process_group_size', type=int, default=defaults['syncbn_process_group_size'], help='SyncBN process group size')
    parser.add_argument('--seed', type=int, default=defaults['seed'], help='Random seed')

    parser.add_argument('--cvae_loss_scaler', type=float, default=defaults['cvae_loss_scaler'])
    parser.add_argument('--prot_decoding_loss_scaler', type=float, default=defaults['prot_decoding_loss_scaler'])
    
    # Optional fields
    parser.add_argument('--model', type=str, default=defaults['model'], help='Model to use')
    parser.add_argument('--optimizer', type=str, default=defaults['optimizer'], help='Optimizer to use')
    parser.add_argument('--lr_schedule', type=str, default=defaults['lr_schedule'], help='Learning rate schedule')
    parser.add_argument('--queue', type=str, default=defaults['queue'], help='Queue')
    parser.add_argument('--train_loader', type=str, default=defaults['train_loader'], help='Training loader')
    parser.add_argument('--training_stats', type=str, default=defaults['training_stats'], help='Training statistics')
    parser.add_argument('--condition_key', type=str, default=defaults['condition_key'], help='Condition key')
    parser.add_argument('--cell_type_key', type=str, default=defaults['cell_type_key'], help='Cell type key')
    parser.add_argument('--device', type=str, default=defaults['device'], help='Device to use')
    parser.add_argument('--debug', type=bool, default=defaults['debug'], help='Debug mode')
    parser.add_argument('--freezable_prototypes', type=bool, default=defaults['freezable_prototypes'], help='Freezable prototypes')

    return parser