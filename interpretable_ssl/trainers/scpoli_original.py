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
    def __init__(self, parser=None, **kwargs):
        super().__init__(parser, **kwargs)
        print('new')
        if not self.debug:
            self.init_wandb(self.get_model_path(), 0, 0)
        # self.experiment_name = "original-scpoli"
        # self.model_name_version = 2
        self.create_dump_path()

    def get_model(self, adata):
        condition_key = "study"
        cell_type_key = "cell_type"
        return scPoli(
            adata=adata,
            condition_keys=condition_key,
            cell_type_keys=self.cell_type_key,
            latent_dim=self.latent_dims,
            recon_loss="nb",
        )

    def train(self):
        epochs = self.fine_tuning_epochs + self.pretraining_epochs
        model = self.get_model(self.ref.adata)
        model.train(
            n_epochs=epochs,
            pretraining_epochs=self.pretraining_epochs,
            eta=5,
        )
        model_path = self.get_model_path()
        utils.save_model_checkpoint(
            model.model,
            epochs,
            model_path,
        )

    def get_model_path(self):
        return self.get_dump_path() + "/model.pth"

    def get_model_name(self):

        if self.model_name_version < 3:
            # saving model at /home/icb/fatemehs.hashemig/models//pbmc-immune/scpoli-original-latent_dim8.pth
            name = f"scpoli-original-latent_dim{self.latent_dims}"
            return self.append_batch(name)
        elif self.model_name_version < 5:
            return super().get_model_name()
        else:
            if self.experiment_name is None:
                self.experiment_name = 'scpoli'
            return f'{self.experiment_name}_latent{self.latent_dims}_bs{self.batch_size}'

    def load_model(self):
        model = self.get_model(self.ref.adata)
        path = self.get_model_path()
        model.model.load_state_dict(torch.load(path)["model_state_dict"])
        return model

    def load_query_model(self):
        model = self.load_model()
        scpoli_query = scPoli.load_query_data(
            adata=self.query.adata,
            reference_model=model,
            labeled_indices=[],
        )
        return scpoli_query

    def finetune_query_model(self, model, epochs=100):
        model.train(
            n_epochs=self.fine_tuning_epochs,
            pretraining_epochs=self.get_pretraining_epochs(epochs),
            eta=10,
        )
        utils.save_model(
            model.model,
            self.get_query_model_path(),
        )
        return model

    def get_query_model_latent(self, model, adata):
        return model.get_latent(adata, mean=True)

    def encode_batch(self, model, batch):
        batch = self.move_input_on_device(batch)
        scpoli_model = self.get_scpoli_model(model)
        scpoli_model.to(self.device)
        scpoli_model.eval()
        with torch.no_grad():
            x, _, _, _ = scpoli_model(**batch)
        return x

    def get_scpoli_model(self, pretrained_model):
        return pretrained_model.model
