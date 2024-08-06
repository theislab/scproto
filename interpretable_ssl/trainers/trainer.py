import os
import sys
import torch
import wandb
import pandas as pd
import torch.optim as optim
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from itertools import combinations
from torch.utils.data import DataLoader
from interpretable_ssl.models import ssl
from interpretable_ssl.utils import (
    get_device,
    add_prefix_key,
    save_model_checkpoint,
    get_model_dir,
    tensor_to_numpy,
)
from interpretable_ssl.datasets.dataset import SingleCellDataset
from interpretable_ssl.datasets.immune import ImmuneDataset
from interpretable_ssl.evaluation.scib_metrics import calculate_scib_metrics
from interpretable_ssl.trainers.base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = get_device()
        self.dataset = self.get_dataset(self.dataset_id)
        self.input_dim = self.dataset.x_dim
        self.ref, self.query = self.dataset.get_train_test()

    def calculate_ref_query_latent(self):
        pass

    def set_fold(self, fold):
        self.fold = fold
        self.ref_latent, self.query_latent, self.all_latent = None, None, None

    def get_ref_query_latent(self):
        if self.ref_latent is None:
            self.ref_latent, self.query_latent, self.all_latent = (
                self.calculate_ref_query_latent()
            )
        return self.ref_latent, self.query_latent, self.all_latent

    def get_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=0.0005)

    def load_model(self):
        model = self.get_model()
        path = self.get_model_path()
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model, checkpoint["train_indices"], checkpoint["test_indices"]

    def get_dataset(self, dataset_id) -> SingleCellDataset:
        if dataset_id == "pbmc-immune":
            return ImmuneDataset()
        else:
            print("dataset not implemented")
            return None



    def get_model(self):
        pass





    def get_train_test_loader(self):
        train, test = self.dataset.get_train_test()
        print("train dataset size:", len(train))
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def ssl_train_step(self, model, train_loader, optimizer):
        return ssl.train_step(model, train_loader, optimizer, self.device)

    def ssl_test_step(self, model, test_loader):
        return ssl.test_step(model, test_loader, self.device)

    def train_step(self, model, train_loader, optimizer):
        return self.ssl_train_step(model, train_loader, optimizer)

    def test_step(self, model, test_loader):
        return self.ssl_test_step(model, test_loader)

    def init_wandb(self, model_path, train_size, test_size):
        wandb.init(
            project="interpretable-ssl",
            config={
                "num_prototypes": self.num_prototypes,
                "hidden dim": self.hidden_dim,
                "latent_dims": self.latent_dims,
                "device": self.device,
                "model path": model_path,
                "dataset": self.dataset,
                "train size": train_size,
                "test size": test_size,
                "batch size": self.batch_size,
                "description": self.description,
            },
        )

    def log_loss(self, train_loss, test_loss):
        train_loss_dict = add_prefix_key(train_loss.__dict__, "train")
        test_loss_dict = add_prefix_key(test_loss.__dict__, "test")
        train_loss_dict.update(test_loss_dict)
        wandb.log(train_loss_dict)

    def train(self, epochs=100):
        train_loader, test_loader = self.get_train_test_loader()
        model = self.get_model()
        optimizer = self.get_optimizer(model)
        model_path = self.get_model_path()
        self.init_wandb(model_path, len(train_loader.dataset), len(test_loader.dataset))

        best_test_loss = sys.maxsize
        print("start training")
        for epoch in tqdm(range(epochs)):
            train_loss = self.train_step(model, train_loader, optimizer)
            test_loss = self.test_step(model, test_loader)
            self.log_loss(train_loss, test_loss)
            if test_loss.overal < best_test_loss:
                save_model_checkpoint(model, epoch, model_path)

    def get_ssl_representations(self):
        model = self.load_model()
        model.to(self.device)
        x = torch.tensor(self.dataset.adata.X.toarray(), device=self.device)
        latent, _, prot_dist = model.prototype_model(x)
        projections = model.barlow_model.projector(prot_dist)
        return (
            tensor_to_numpy(latent),
            tensor_to_numpy(prot_dist),
            tensor_to_numpy(projections),
        )

    def get_kfold_obj(self, n_splits):
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def train_kfold_cross_val(self, epochs, n_splits=5):
        print("running kfold")
        study_ids = self.dataset.get_study_ids()
        kf = self.get_kfold_obj(n_splits)
        self.fold = 0
        for train_study_index, test_study_index in kf.split(study_ids):
            self.ref, self.query = self.dataset.get_fold_train_test(
                train_study_index, test_study_index
            )
            self.fold += 1
            model_path = self.get_model_path()
            if os.path.exists(model_path):
                print(model_path, " exists")
                continue
            self.train(epochs)

    def evaluate_kfold_models(self, evaluation_fn, n_splits=5):
        study_ids = self.dataset.get_study_ids()
        kf = self.get_kfold_obj(n_splits)
        self.fold = 0
        overall_df = pd.DataFrame()
        for train_study_index, test_study_index in kf.split(study_ids):
            self.ref, self.query = self.dataset.get_fold_train_test(
                train_study_index, test_study_index
            )
            self.fold += 1
            print(f"Evaluating fold {self.fold}")
            fold_df = evaluation_fn()
            fold_df["fold"] = self.fold
            overall_df = overall_df.append(fold_df, ignore_index=True)
        return overall_df

    def scib_metrics_all(self):
        _, _, latent = self.get_ref_query_latent()
        df, _ = calculate_scib_metrics(self.dataset.adata, latent)
        return df

    def query_scib_metrics(self):
        _, query, _ = self.get_ref_query_latent()
        df, _ = calculate_scib_metrics(self.query.adata, query)
        return df

    def train_custom_cross_val(self, epochs):
        self.custom_cross_val = True
        print("running custom cross validation")
        study_ids = self.dataset.get_study_ids()
        num_studies = len(study_ids)
        test_combinations = list(combinations(range(num_studies), 2))
        self.fold = 0
        for test_study_indices in test_combinations:
            train_study_indices = [
                i for i in range(num_studies) if i not in test_study_indices
            ]
            test_study_indices = list(test_study_indices)
            self.ref, self.query = self.dataset.get_fold_train_test(
                train_study_indices, test_study_indices
            )
