import os
import wandb
from interpretable_ssl.utils import get_device
from interpretable_ssl.datasets.immune import ImmuneDataset
from interpretable_ssl.datasets.hlca import HLCADataset
from interpretable_ssl.trainers.base import TrainerBase
from interpretable_ssl.datasets.pancreas import PancreasDataset
from interpretable_ssl.utils import log_time


class Trainer(TrainerBase):
    # @log_time('trainer')
    def __init__(self, debug=False, dataset=None, ref_query=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = get_device()
        self.dataset = dataset
        if self.dataset is None:
            print(f"dataset is None, loading {self.dataset_id}")
            if self.no_data == "False":
                self.dataset = self.get_dataset(self.dataset_id)
        self.input_dim = self.dataset.x_dim

        if ref_query is None:
            self.ref, self.query = self.dataset.get_train_test()
        else:
            self.ref, self.query = ref_query
        self.debug = debug
        self.ref_latent, self.query_latent, self.all_latent = None, None, None

        if (not self.debug) and (self.wandb_sweep != 1):
            self.init_wandb()

    def get_model(self):
        pass

    def get_dataset(self, dataset_id):

        if dataset_id == "pbmc-immune":
            return ImmuneDataset()
        if dataset_id == "hlca":
            return HLCADataset()
        elif dataset_id == 'pancreas':
            return PancreasDataset()
        else:
            print("dataset not implemented")
            return None

    def init_wandb(self, path=None):
        if path is None:
            path = self.get_dump_path()
        set_job_name = (self.job_name is None) or (self.job_name == "")
        if set_job_name:
            self.job_name = f"{self.get_model_name()}/{self.dataset}"
        wandb.init(
            name=self.job_name,
            # project="interpretable-ssl",
            config={
                "num_prototypes": self.num_prototypes,
                "hidden dim": self.hidden_dim,
                "latent_dims": self.latent_dims,
                "device": self.device,
                "model path": path,
                "dataset": self.dataset,
                "train size": len(self.ref),
                "test size": len(self.query),
                "batch size": self.batch_size,
            },
        )
