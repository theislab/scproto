from pathlib import Path
from interpretable_ssl.configs.defaults import *


def get_model_dir():
    return "/home/icb/fatemehs.hashemig/models/"


class TrainerBase:
    def __init__(
        self,
        dataset_id="immune",
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
        epochs=300,
        linear_eval=False,
        only_eval=False,
        use_early_stopping=False,
        debug=False,
    ) -> None:
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
        self.epochs = epochs
        self.linear_eval = linear_eval
        self.only_eval = only_eval
        self.use_early_stopping = use_early_stopping
        self.debug = debug

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

class ScpoliTrainerBase(TrainerBase):
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
class SwavTrainerBase(TrainerBase):
    def __init__(self, parser=None, **kwargs):
        # Get default values for Swav
        self.default_values = get_swav_defaults()
        
        kwargs = self.update_kwargs(parser, kwargs)
        
        # Extract SwavTrainer-specific arguments
        scpoli_keys = get_scpoli_defaults().keys()
        scpoli_kwargs = {key: kwargs.pop(key) for key in scpoli_keys if key in kwargs}
        super().__init__(**scpoli_kwargs)
        
        # Set specific attributes for SwavTrainer
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.nmb_prototypes = self.num_prototypes
        self.set_experiment_name()
        self.create_dump_path()
        self.use_projector_out = False