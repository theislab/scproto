from pathlib import Path
from interpretable_ssl.configs.defaults import *
from interpretable_ssl.configs.constants import *
import os
from constants import *
from interpretable_ssl.utils import log_time
from interpretable_ssl.model_name import generate_model_name

from scarches.dataset.scpoli.anndata import MultiConditionAnnotatedDataset
import scarches.trainers.scpoli._utils as scpoli_utils
from torch.utils.data import DataLoader


class TrainerBase:
    # @log_time('trainer base')
    def __init__(self, **kwargs) -> None:
        # Get the default values from the function
        defaults = get_defaults().copy()
        self.default_values = get_defaults().copy()
        # Update defaults with any provided keyword arguments
        defaults.update(kwargs)

        # Assign each default value to an instance variable
        for key, value in defaults.items():
            setattr(self, key, value)
        
        
        self.set_experiment_name()
        self.params = self.__dict__.copy()
        self.create_dump_path()
        self.create_temp_res_path()
        # if self.training_type != "semi_supervised" and self.training_type != "fully_supervised":
        #     self.pretraining_epochs += self.fine_tuning_epochs

    def get_metric_file_path(self, split):
        if self.model_name_version == 3:
            base = f"{split}-scib"
        elif self.model_name_version >= 3.5:
            base = f"{split}-metrics"
        if self.finetuning:
            base = f"{base}-semi-supervised"
        filename = f"{base}.csv"
        return os.path.join(self.get_dump_path(), filename)

    def check_scib_metrics_exist(self):
        path = self.get_metric_file_path("ref")
        if os.path.exists(path):
            print(path, " exists")
            return True

        name = "semi-supervised" if self.finetuning else ""
        return any(
            name in file and file.endswith(".csv")
            for _, _, files in os.walk(self.get_dump_path())
            for file in files
        )

    def create_dump_path(self):
        self.dump_path = self.get_dump_path()
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)

    def create_temp_res_path(self):
        temp_res_path = self.get_temp_res_path()
        if self.save_temp_res == 1 and not os.path.exists(temp_res_path):
            os.makedirs(temp_res_path)

    def set_experiment_name(self):
        if self.experiment_name == "":
            self.experiment_name = None
        if self.experiment_name is not None:
            return

        if self.dump_name_version >= 4:
            self.experiment_name = f"swav"
        else:
            self.set_old_experiment_name()

    def set_old_experiment_name(self):
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

    def generate_name_based_on_changes(self):
        return generate_model_name(get_defaults().copy(), self.params)

    def get_model_name(self):
        if self.model_name_version >= 5:
            return self.generate_name_based_on_changes()
        else:
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
            if self.model_name_version >= 3:
                base = self.append_batch(base)

            if self.training_type == "semi_supervised":
                base = f"{base}-semi"
            return base

    def append_batch(self, base):
        if self.is_swav == 1 and (self.batch_size == self.default_values["batch_size"]):
            return base
        else:
            return f"{base}-bs{self.batch_size}"

    def get_save_dir(self):
        if self.training_type == "transfer_learning":
            return f"{MODEL_DIR}/{self.pretrain_dataset_id}_{self.finetune_dataset_id}/"
        return f"{MODEL_DIR}/{self.dataset_id}/"

    def get_swav_dump_path(self):
        dump_path = self.get_general_dump_path()

        if self.dump_name_version != 1 and self.dump_name_version < 4:
            dump_path = f"{dump_path}_aug{self.nmb_crops[0]}_latent{self.latent_dims}"

        if self.dump_name_version > 2 and self.dump_name_version < 4:
            dump_path = f"{dump_path}_aug-type-{self.augmentation_type}"

        if self.dump_name_version > 3:
            dump_path = f"{dump_path}_aug-{self.augmentation_type}{self.nmb_crops[0]}"
            dump_path = self.add_additional_parameters(dump_path)

        return dump_path

    def get_temp_res_path(self):
        return f"{self.temp_res_path}/{self.get_model_name()}/"

    def get_general_dump_path(self):
        name = self.get_model_name()
        save_dir = self.get_save_dir()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return f"{save_dir}{name}"

    def get_dump_path(self):
        if self.is_swav == 1 and self.model_name_version < 5:
            return self.get_swav_dump_path()
        else:
            return self.get_general_dump_path()

    def get_model_path(self):
        return self.get_dump_path() + ".pth"

    def get_abbreviation(self, key):

        if self.model_name_version < 4:
            return key

        if key in ABBREVIATIONS:
            return ABBREVIATIONS[key]
        return key

    def add_additional_parameters(self, dump_path):
        """
        Modify the dump_path based on specific attributes and a list of keys
        if their current values differ from the default values.

        Parameters
        ----------
        dump_path : str
            The base path to modify.
        keys_to_check : list or None
            List of keys to check against their default values. If None, no additional keys are checked.

        Returns
        -------
        str
            The modified dump_path with additional parameters included if their values differ from the default.
        """
        # Preserve current functionality
        dump_path = f"{dump_path}_ep{self.epsilon}"
        if self.use_projector:
            dump_path = (
                f"{dump_path}_use-projector_hmlp{self.hidden_mlp}_sdim{self.swav_dim}"
            )
        if self.default_values["model_version"] != self.model_version:
            dump_path = f"{dump_path}_model-v{self.model_version}"
        if self.default_values["longest_path"] != self.longest_path:
            dump_path = f"{dump_path}_lp{self.longest_path}"

        keys_to_check = [
            "dimensionality_reduction",
            "k_neighbors",
            "freeze_prototypes_niters",
            "temperature",
            "epsilon",
            "prot_init",
            "propagation_reg",
            "prot_emb_sim_reg",
            "loss_type",
        ]
        # Check additional keys, if provided
        if keys_to_check:
            for key in keys_to_check:
                if key in self.default_values:
                    current_value = getattr(self, key, None)
                    default_value = self.default_values[key]
                    if current_value != default_value:
                        dump_path = (
                            f"{dump_path}_{self.get_abbreviation(key)}-{current_value}"
                        )

        return dump_path

    def prepare_scpoli_dataloader(self, adata, scpoli_model, shuffle=True):

        if "condition_combined" not in adata.obs:
            adata.obs["conditions_combined"] = adata.obs[[self.condition_key]].apply(
                lambda x: "_".join(x), axis=1
            )
        dataset = MultiConditionAnnotatedDataset(
            adata,
            condition_keys=[self.condition_key],
            # cell_type_keys=[self.cell_type_key],
            condition_encoders=scpoli_model.condition_encoders,
            conditions_combined_encoder=scpoli_model.conditions_combined_encoder,
            # cell_type_encoder=scpoli_model.cell_type_encoder,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=scpoli_utils.custom_collate,
            shuffle=shuffle,
        )
        return loader
