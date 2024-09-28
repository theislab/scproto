from pathlib import Path
from interpretable_ssl.configs.defaults import *
import os


def get_model_dir():
    return "/home/icb/fatemehs.hashemig/models/"


class TrainerBase:
    def __init__(
        self,
        dataset_id="pbmc-immune",
        model_name_version=3,
        num_prototypes=8,
        hidden_dim=64,
        latent_dims=8,
        batch_size_version=2,
        batch_size=None,
        fine_tuning_epochs=None,
        custom_cross_val=False,
        description=None,
        experiment_name=None,
        # epochs=300,
        pretraining_epochs=300,
        linear_eval=False,
        only_eval=False,
        use_early_stopping=False,
        debug=False,
        training_type='pretrain',
    ) -> None:
        print("new base init")
        self.num_prototypes = num_prototypes
        self.hidden_dim = hidden_dim
        self.latent_dims = latent_dims
        self.batch_size_version = batch_size_version
        self.batch_size = (
            batch_size
            if batch_size is not None
            else (512 if self.batch_size_version == 2 else 64)
        )
        self.dataset_id = dataset_id

        self.description = description
        self.experiment_name = experiment_name
        self.fold = None

        self.fine_tuning_epochs = fine_tuning_epochs
        self.custom_cross_val = custom_cross_val
        self.model_name_version = model_name_version
        self.ref_latent, self.query_latent, self.all_latent = None, None, None
        self.pretraining_epochs = pretraining_epochs
        self.linear_eval = linear_eval
        self.only_eval = only_eval
        self.use_early_stopping = use_early_stopping
        self.debug = debug
        self.training_type = training_type
        if self.training_type == 'semi_supervised':
            self.semi_supervised = True
        else:
            self.semi_supervised = False

    def append_batch(self, name):
        if self.model_name_version == 2:
            name = f"{name}_bs{self.batch_size}"
        return name

    def get_model_name(self):
        base = f"num-prot-{self.num_prototypes}"
        if self.model_name_version < 3:
            base += f"_hidden-{self.hidden_dim}_bs-{self.batch_size}"
        else:
            base += f"_latent{self.latent_dims}"

        if self.experiment_name is not None:
            if self.model_name_version < 3:
                base = f"{self.experiment_name}-{base}"
            else:
                base = f"{self.experiment_name}_{base}"
        if self.model_name_version > 3:
            base = f"{base}-bs{self.batch_size}"

        if self.semi_supervised:
            base = f"{base}-semi"
        return base

    def get_dump_path(self):
        name = self.get_model_name()
        save_dir = f"{get_model_dir()}/{self.dataset_id}/"
        if self.fold:
            save_dir += f"{name}/"
            if self.custom_cross_val:
                save_dir = f"{save_dir[:-1]}_ccross-val/"
            name = f"fold-{self.fold}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return f"{save_dir}{name}"

    def get_model_path(self):
        return self.get_dump_path() + ".pth"


from interpretable_ssl.configs.defaults import *


class TrainerBaseV2:
    def __init__(self, parser=None, **kwargs):
        # Get default values for Swav
        self.default_values = get_merged_defaults()
        kwargs = self.update_kwargs(parser, kwargs)

        # Set specific attributes for SwavTrainer
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.default_values = get_merged_defaults()
        self.nmb_prototypes = self.num_prototypes
        self.set_experiment_name()
        self.create_dump_path()

    def update_kwargs(self, parser, kwargs):
        if parser is not None:
            parser = self.add_parser_args(parser)
            args = parser.parse_args()
            kwargs.update(vars(args))

        # Use default values for any missing kwargs
        for key, value in self.default_values.items():
            if value == "":
                value = None
            kwargs.setdefault(key, value)
        return kwargs

    def add_parser_args(self, parser):
        # Add arguments to parser with default values from dictionary
        for key, value in self.default_values.items():
            if isinstance(value, bool):
                # Handle boolean arguments with action='store_true'
                parser.add_argument(
                    f"--{key}",
                    action="store_true",
                    help=f"Set {key} to true (default is {value})",
                )
            else:
                # Handle other types of arguments
                arg_type = type(value) if value is not None else str
                if value == "":
                    value = None
                parser.add_argument(
                    f"--{key}",
                    type=arg_type,
                    default=value,
                    help=f"Set {key} (default is {value})",
                )
        return parser

    def set_experiment_name(self):
        if self.experiment_name is not None:
            return
        if self.dump_name_version < 4:
            return
        if self.prot_decoding_loss_scaler == 0 and self.cvae_loss_scaler == 0:
            self.experiment_name = "swav-only"
        elif self.prot_decoding_loss_scaler > 0 and self.cvae_loss_scaler == 0:
            self.experiment_name = "swav-interpretable"
        elif self.prot_decoding_loss_scaler > 0 and self.cvae_loss_scaler > 0:
            self.experiment_name = "swav-all-loss"
        else:
            # Optionally handle any other case, or set a default value
            self.experiment_name = "swav"
        self.experiment_name = f"{self.experiment_name }_iloss{self.prot_decoding_loss_scaler}_closs{self.cvae_loss_scaler}"

    def create_dump_path(self):
        self.dump_path = self.get_dump_path()
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)

    def get_model_name(self):
        base = f"num-prot-{self.num_prototypes}"
        if self.model_name_version < 3:
            base += f"_hidden-{self.hidden_dim}_bs-{self.batch_size}"
        else:
            base += f"_latent{self.latent_dims}"

        if self.experiment_name is not None:
            if self.model_name_version < 3:
                base = f"{self.experiment_name}-{base}"
            else:
                base = f"{self.experiment_name}_{base}"
        if self.model_name_version > 3:
            base = f"{base}-bs{self.batch_size}"

        return base

    def get_dump_base(self):
        name = self.get_model_name()
        save_dir = f"{get_model_dir()}/{self.dataset_id}/"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return f"{save_dir}{name}"

    def get_dump_path(self):
        dump_path = self.get_dump_base()

        if self.dump_name_version != 1 and self.dump_name_version < 4:
            dump_path = f"{dump_path}_aug{self.nmb_crops[0]}_latent{self.latent_dims}"

        if self.dump_name_version > 2 and self.dump_name_version < 4:
            dump_path = f"{dump_path}_aug-type-{self.augmentation_type}"

        if self.dump_name_version > 3:
            dump_path = f"{dump_path}_aug-{self.augmentation_type}{self.nmb_crops[0]}"
            dump_path = self.add_additional_parameters(dump_path)

        return dump_path

    def add_additional_parameters(self, dump_path):

        # Add use_projector if it is true
        dump_path = f"{dump_path}_ep{self.epsilon}"
        if self.use_projector:
            dump_path = (
                f"{dump_path}_use-projector_hmlp{self.hidden_mlp}_sdim{self.swav_dim}"
            )
        if self.default_values["model_version"] != self.model_version:
            dump_path = f"{dump_path}_model-v{self.model_version}"
        return dump_path
