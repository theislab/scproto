from interpretable_ssl.trainers.trainer import Trainer
from interpretable_ssl.models.scpoli import *
from sklearn.model_selection import train_test_split
from interpretable_ssl.datasets.immune import ImmuneDataset
from interpretable_ssl.train_utils import optimize_model
import sys
from interpretable_ssl.evaluation.visualization import plot_umap
from interpretable_ssl.loss_manager import *
from interpretable_ssl import utils
from torch.utils.data import WeightedRandomSampler
from scarches.models.scpoli import scPoli
import numpy as np
from interpretable_ssl.train_utils import *

class ScpoliTrainer(Trainer):
    def __init__(self, dataset=None) -> None:
        super().__init__(dataset=dataset)

        self.latent_dims = 8
        self.num_prototypes = 16
        self.experiment_name = "scpoli"
        self.train_adata = None
        self.val_adata = None
        self.best_val_loss = sys.maxsize
        self.use_weighted_sampling = False

    def reference_mapping(self, reference_model):
        scpoli_query = scPoli.load_query_data(
            adata=self.query.adata,
            reference_model=reference_model,
            labeled_indices=[],
        )
        scpoli_query.train(
            n_epochs=self.fine_tuning_epochs, pretraining_epochs=40, eta=10
        )
        return scpoli_query

    def get_model_ref_query_latent(self, scpoli_model):

        query_model = self.reference_mapping(scpoli_model)
        query_latent = query_model.get_latent(self.query.adata, mean=True)
        reference_latent = query_model.get_latent(self.ref.adata, mean=True)
        return reference_latent, query_latent

    def get_weighted_sampler(self, adata):
        encoded_labels = adata.obs["encoded_cell_type"]
        unique_labels = encoded_labels.unique()
        label_to_index = {label: index for index, label in enumerate(unique_labels)}

        mapped_labels = encoded_labels.map(label_to_index)
        class_counts = mapped_labels.value_counts().values

        class_weights = np.zeros_like(class_counts, dtype=float)
        non_zero_mask = class_counts != 0
        class_weights[non_zero_mask] = 1.0 / class_counts[non_zero_mask]

        weights = class_weights[mapped_labels]
        return WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

    def split_train_test(self, ref):
        train_idx, val_idx = train_test_split(range(len(ref.adata)))
        train, val = ref.adata[train_idx], ref.adata[val_idx]
        return train, val

    # def prepare_scpoli_data_splits(self, ref, scpoli_model):
    #     train_adata, val_adata = self.split_train_test(ref)
    #     if self.use_weighted_sampling:
    #         sampler = self.get_weighted_sampler(train_adata)
    #     else:
    #         sampler = None
    #     train_loader = generate_scpoli_dataloder(train_adata, scpoli_model, sampler, batch_size=self.batch_size)
    #     val_loader = generate_scpoli_dataloder(val_adata, scpoli_model, batch_size=self.batch_size)
    #     return train_adata, train_loader, val_adata, val_loader

    def to_save(self, val_loss):
        if val_loss.overal < self.best_val_loss:
            self.best_val_loss = val_loss.overal
            return True
        return False

    def train_step(self, model, optimizer):
        return scpoli_train_step(model, self.train_adata, optimizer, self.batch_size)

    def test_step(self, model):
        return scpoli_test_step(model, self.val_adata, self.batch_size)

    def train(self, epochs):
        # prepare data (train, test)
        # define model
        # init training params + wandb
        # train loop:
        #   for each epoch:
        #      calculate loss
        #      optimize
        #      log loss
        print("running scpoli trainer class train")
        ref, query = self.ref, self.query
        model = self.get_model(ref.adata)
        model.to(self.device)

        self.train_adata, self.val_adata = self.split_train_test(ref)

        # init training parameter and wandb
        optimizer = self.get_optimizer(model)
        model_path = self.get_model_path()

        # init wandb
        self.init_wandb(model_path, len(self.train_adata), len(self.val_adata))
        self.best_val_loss = sys.maxsize

        for epoch in range(epochs):

            train_loss = self.train_step(model, optimizer)
            val_loss = self.test_step(model)
            self.log_loss(train_loss, val_loss)
            if self.to_save(val_loss):
                utils.save_model_checkpoint(
                    model,
                    epoch,
                    model_path,
                )
        return train_loss.overal, self.best_val_loss

    def load_model(self):

        model = self.get_model(self.ref.adata)
        path = self.get_model_path()
        model.load_state_dict(torch.load(path)["model_state_dict"])
        return model

    def get_representation(self, model, adata):
        return model.get_representation(adata)

    def visualize(self):
        model = self.load_model()
        model.to(self.device)
        # get latent representation of reference data
        data_latent_source = self.get_representation(model, self.ref.adata)
        self.ref.adata.obsm[f"{self.experiment_name}"] = data_latent_source

        plot_umap(self.ref.adata, f"{self.experiment_name}")

    def get_query_model(self):
        model = self.load_model()
        scpoli_query = scPoli.load_query_data(
            adata=self.query.adata,
            reference_model=model.scpoli,
            labeled_indices=[],
        )
        return scpoli_query
        
    def calculate_ref_query_latent(self, fine_tuning = True):
        scpoli_query = self.get_query_model()
        if fine_tuning:
            scpoli_query.train(n_epochs=self.fine_tuning_epochs, pretraining_epochs=40, eta=10)
        query_latent = scpoli_query.get_latent(self.query.adata, mean=True)
        ref_latent = scpoli_query.get_latent(self.ref.adata, mean=True)
        all_latent = scpoli_query.get_latent(self.dataset.adata, mean = True)
        return ref_latent, query_latent, all_latent


class ScpoliProtBarlowTrainer(ScpoliTrainer):
    def __init__(self, dataset, projection_version=0) -> None:
        super().__init__(dataset)
        self.experiment_name = 'barlow'
        self.projection_version = projection_version
    def get_model_name(self):
        name = super().get_model_name()
        if self.projection_version != 0:
            name = f'{name}_projection-version-{self.projection_version}'
        return name
    def get_model(self, adata):
        return BarlowPrototypeScpoli(adata, self.latent_dims, self.num_prototypes, self.projection_version)


class LinearTrainer(ScpoliTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.num_prototypes = 32
        self.experiment_name = "linear-prot-scpoli-task_ratio10"

    def get_model(self, adata):
        head = nn.Linear(self.num_prototypes, self.dataset.num_classes, bias=False)
        return LinearPrototypeScpoli(adata, self.latent_dims, self.num_prototypes, head)

    def train_step(self, model, train_loader, optimizer):
        total_loss = PrototypeLoss()
        for scpoli_batch in train_loader:
            batch_loss = model(scpoli_batch)
            total_loss += batch_loss
            optimize_model(batch_loss, optimizer)
        total_loss.normalize(len(train_loader))
        return total_loss

    def test_step(self, model, test_loader):
        model.eval()
        test_loss = PrototypeLoss()
        with torch.inference_mode():
            for scpoli_batch in test_loader:
                test_loss += model(scpoli_batch)
        test_loss.normalize(len(test_loader))
        return test_loss


class BarlowTrainer(ScpoliTrainer):
    def __init__(self) -> None:
        print("running barlow-scpoli")
        super().__init__()
        self.experiment_name = "barlow-scpoli"

    def get_model(self, adata):
        return BarlowScpoli(adata, self.latent_dims)


class ScpoliOriginal(ScpoliTrainer):

    def __init__(self) -> None:
        super().__init__()
        condition_key = "study"
        cell_type_key = "cell_type"
        self.scpoli_trainer = scPoli(
            adata=self.ref.adata,
            condition_keys=condition_key,
            cell_type_keys=cell_type_key,
            latent_dim=self.latent_dims,
            recon_loss="nb",
        )

    def get_model(self, adata):

        return self.scpoli_trainer.model

    def train(self):
        self.scpoli_trainer.train(
            n_epochs=100,
            pretraining_epochs=50,
        )
        path = self.get_model_path()
        utils.save_model(self.scpoli_trainer.model, path)

    def get_representation(self, model, adata):
        return self.scpoli_trainer.get_latent(adata, mean=True)

    def get_model_name(self):
        return f"original-scpoli-latent{self.latent_dims}"


class SimClrTrainer(ScpoliTrainer):
    def __init__(self, dataset=None) -> None:
        super().__init__(dataset)
        self.experiment_name = 'simclr'
    def get_model(self, adata):
        return SimClrPrototype(adata, self.latent_dims, self.num_prototypes)