# load data
# define model
# set training parameters
# train loop

from interpretable_ssl.datasets.immune import ImmuneDataset
from scarches.models.scpoli import scPoli
from interpretable_ssl import utils
from interpretable_ssl.trainers.scpoli_trainer import ScpoliTrainer
import torch
import wandb
import sys


class OriginalTrainer(ScpoliTrainer):
    def __init__(self, dataset=None) -> None:
        super().__init__(dataset=dataset)
        self.experiment_name = "original-scpoli"

    def get_model(self, adata):
        condition_key = "study"
        cell_type_key = "cell_type"
        return scPoli(
            adata=adata,
            condition_keys=condition_key,
            cell_type_keys=cell_type_key,
            latent_dim=self.latent_dims,
            recon_loss="nb",
        )

    def train(self, epochs):
        pretraining_epochs = 40
        model = self.get_model(self.ref.adata)
        model.train(
            n_epochs=epochs,
            pretraining_epochs=pretraining_epochs,
            eta=5,
        )
        model_path = self.get_model_path()
        utils.save_model_checkpoint(
            model.model,
            "",
            epochs,
            model_path,
            self.train_study_index,
            self.test_study_index,
        )

    def get_model_name(self):
        return f"scpoli-original-latent_dim{self.latent_dims}"

    def load_model(self):
        model = self.get_model(self.ref.adata)
        path = self.get_model_path()
        model.model.load_state_dict(torch.load(path)["model_state_dict"])
        return model
    
    
    def get_query_model(self):
        model = self.load_model()
        scpoli_query = scPoli.load_query_data(
            adata=self.query.adata,
            reference_model=model,
            labeled_indices=[],
        )
        return scpoli_query