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
from interpretable_ssl.trainers.adaptive_trainer import AdoptiveTrainer


class OriginalTrainer(AdoptiveTrainer):
    def __init__(
        self, debug=False, dataset=None, ref_query=None, parser=None, **kwargs
    ):
        self.is_swav = 0
        super().__init__(debug, dataset, ref_query, parser, **kwargs)
        print("input data size: ", len(self.ref))
        self.model = self.get_model()
        
        # if not self.debug:
        #     self.init_wandb()
        # self.experiment_name = "original-scpoli"
        # self.model_name_version = 2
        # self.create_dump_path()

    def set_experiment_name(self):
        if self.model_name_version < 7:
            return super().set_experiment_name()
        
        if self.experiment_name is not None:
            if self.experiment_name != "":
                return
        else:
            self.experiment_name = "scpoli"

    def get_model(self):

        adata = self.ref.adata
        if self.training_type == "pretrain":
            cell_type_key = None
        else:
            cell_type_key = "cell_type"
        condition_key = "study"

        return scPoli(
            adata=adata,
            condition_keys=condition_key,
            cell_type_keys=cell_type_key,
            latent_dim=self.latent_dims,
            recon_loss=self.recon_loss,
        )

    def train(self, pretrain=True, finetune=True):

        if pretrain and finetune:
            epochs = self.fine_tuning_epochs + self.pretraining_epochs

        if pretrain and not (finetune):
            epochs = self.pretraining_epochs

        pretraining_epochs = self.pretraining_epochs
        if finetune and not (pretrain):
            epochs = self.fine_tuning_epochs
            pretraining_epochs = 0

        self.model.train(
            n_epochs=epochs,
            pretraining_epochs=pretraining_epochs,
            eta=5,
        )
        model_path = self.get_model_path()
        utils.save_model_checkpoint(
            self.model.model,
            epochs,
            model_path,
        )
        self.save_metrics()

    def get_model_path(self):
        return self.get_dump_path() + "/model.pth"

    def get_model_name(self):

        if self.model_name_version < 3:
            # saving model at /home/icb/fatemehs.hashemig/models//pbmc-immune/scpoli-original-latent_dim8.pth
            name = f"scpoli-original-latent_dim{self.latent_dims}"
            return self.append_batch(name)
        # elif self.model_name_version < 5:
        #     return super().get_model_name()
        elif self.model_name_version < 7:
            if self.experiment_name is None:
                self.experiment_name = "scpoli"
            if self.model_name_version < 5:
                return f"{self.experiment_name}_latent{self.latent_dims}_bs{self.batch_size}"
            else:
                return f"{self.experiment_name}_training-{self.training_type}"
        else:
            return super().get_model_name()

    def load_model(self):
        model = self.get_model()
        path = self.get_model_path()
        model.model.load_state_dict(torch.load(path)["model_state_dict"])
        return model

    def adapt_ref_model(self, ref_model, adata, retrain_epochs=0):
        query_model = self.get_model()
        query_model.model.load_state_dict(ref_model.model.state_dict())
        scpoli_query = scPoli.load_query_data(
            adata=adata,
            reference_model=query_model,
            labeled_indices=[],
        )
        if retrain_epochs > 0:
            scpoli_query.train(n_epochs=retrain_epochs, 
                              pretraining_epochs=retrain_epochs)
        return scpoli_query
            

    def load_query_model(self, adata=None):
        if adata is None:
            adata = self.query.adata
        model = self.load_model()
        scpoli_query = scPoli.load_query_data(
            adata=adata,
            reference_model=model,
            labeled_indices=[],
        )
        return scpoli_query

    def finetune_query_model(self, model):
        model.train(
            n_epochs=self.fine_tuning_epochs,
            pretraining_epochs=self.pretraining_epochs,
            eta=10,
        )
        utils.save_model(
            model.model,
            self.get_query_model_path(),
        )
        return model

    def get_query_model_latent(self, model, adata):
        return model.get_latent(adata, mean=True)

    def encode_batch(self, model, batch, return_maped=False, return_mapped_idx=None):
        batch = self.move_input_on_device(batch)
        scpoli_model = self.get_scpoli(model)
        scpoli_model.to(self.device)
        scpoli_model.eval()
        with torch.no_grad():
            x, _, _, _ = scpoli_model(**batch)
        return x

    def get_scpoli(self, pretrained_model, return_model=True):
        if return_model:
            return pretrained_model.model
        return pretrained_model

    def train_semi_supervised(self):
        self.split_train_data()
        self.ref = self.partial_ref
        self.setup()
        self.train(pretrain=True, finetune=False)
        self.finetuning = True
        self.ref = self.finetune_ds
        self.train(pretrain=False, finetune=True)

    def transfer_learning(self):
        self.dataset = self.get_dataset(self.pretrain_dataset_id)
        self.ref, self.query = self.dataset.get_train_test()
        self.train(pretrain=True, finetune=False)

        # finetune
        self.finetuning = True
        self.dataset = self.get_dataset(self.finetune_dataset_id)
        self.ref, self.query = self.dataset.get_train_test()
        model = self.load_query_model(self.ref)
        self.train(pretrain=False, finetune=True)

    def train_fully_supervised(self):
        self.finetuning = True
        self.train(pretrain=True, finetune=True)
