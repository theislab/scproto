import interpretable_ssl.utils as utils
import torch.optim as optim
import wandb
from tqdm.auto import tqdm
from interpretable_ssl.datasets.dataset import SingleCellDataset
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from interpretable_ssl.models import ssl
import torch
from sklearn.model_selection import KFold
import os
from copy import deepcopy
from interpretable_ssl.evaluation.scib_metrics import *
import pandas as pd
from interpretable_ssl.trainers.linear import *
from itertools import combinations
import os

class Trainer:
    def __init__(
        self, partially_train_ratio=None, self_supervised=False, dataset=None
    ) -> None:
        self.num_prototypes = 8
        self.hidden_dim, self.latent_dims = 64, 8
        self.batch_size_version = 2
        self.batch_size = 64

        if self.batch_size_version == 2:
            self.batch_size = 512

        self.device = utils.get_device()
        if dataset:
            self.dataset = dataset
        else:
            self.dataset = self.get_dataset()
        self.self_supervised = self_supervised

        self.input_dim = self.dataset.x_dim

        self.partially_train_ratio = partially_train_ratio
        self.description = None
        self.experiment_name = None
        self.fold = None
        self.ref, self.query = self.dataset.get_train_test()
        self.fine_tuning_epochs = None
        # only works when using k fold cross val
        self.train_study_index, self.test_study_index = None, None
        self.custom_cross_val = False
        self.model_name_version = 1
        self.ref_latent, self.query_latent, self.all_latent = None, None, None

    def append_batch(self, name):
        if self.model_name_version == 2:
            name = f"{name}_bs{self.batch_size}"
        return name

    def evaluate_classification(self):
        pass

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

    def get_dataset(self) -> SingleCellDataset:
        pass

    def get_model_name(self):
        base = f"num-prot-{self.num_prototypes}_hidden-{self.hidden_dim}_bs-{self.batch_size}"

        if self.experiment_name:
            base = self.experiment_name + "-" + base

        if self.self_supervised:
            base = f"ssl-{base}"

        if self.partially_train_ratio:
            base = f"{base}_train-ratio-{self.partially_train_ratio}"

        return base

    def get_model(self):
        pass

    def get_model_path(self):
        name = self.get_model_name()
        save_dir = utils.get_model_dir() + f"/{self.dataset.name}/"
        if self.fold:
            save_dir = f"{save_dir}/{name}/"
            if self.custom_cross_val:
                save_dir = f"{save_dir[:-1]}_ccross-val/"
            name = f"fold-{self.fold}"
        Path.mkdir(Path(save_dir), exist_ok=True)
        return save_dir + name + ".pth"

    def get_train_test_loader(self):
        train, test = self.dataset.get_train_test()
        if self.partially_train_ratio:
            train = utils.sample_dataset(train, self.partially_train_ratio)
            test = utils.sample_dataset(test, self.partially_train_ratio)
        print("train dataset size: ", len(train))
        train_loader, test_loader = DataLoader(
            train, batch_size=self.batch_size, shuffle=True
        ), DataLoader(test, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def ssl_train_step(self, model, train_loader, optimizer):
        return ssl.train_step(model, train_loader, optimizer, self.device)

    def ssl_test_step(self, model, test_loader):
        return ssl.test_step(model, test_loader, self.device)

    def calssification_train_step(self, model, train_loader, optimizer):
        pass

    def classification_test_step(self, model, test_loader):
        pass

    def train_step(self, model, train_loader, optimizer):
        if self.self_supervised:
            return self.ssl_train_step(model, train_loader, optimizer)
        else:
            return self.calssification_train_step(model, train_loader, optimizer)

    def test_step(self, model, test_loader):
        if self.self_supervised:
            return self.ssl_test_step(model, test_loader)
        else:
            return self.classification_test_step(model, test_loader)

    def init_wandb(self, model_path, train_size, test_size):

        wandb.init(
            # set the wandb project where this run will be logged
            project="interpretable-ssl",
            # track hyperparameters and run metadata
            config={
                "num_prototypes": self.num_prototypes,
                "hidden dim": self.hidden_dim,
                "latent_dims": self.latent_dims,
                "device": self.device,
                "model path": model_path,
                "dataset": self.dataset,
                "partially train ratio": self.partially_train_ratio,
                "train size": train_size,
                "test size": test_size,
                "batch size": self.batch_size,
                "description": self.description,
                "self-supervised": self.self_supervised,
            },
        )

    def log_loss(self, train_loss, test_loss):

        train_loss_dict = utils.add_prefix_key(train_loss.__dict__, "train")

        test_loss_dict = utils.add_prefix_key(test_loss.__dict__, "test")

        train_loss_dict.update(test_loss_dict)

        wandb.log(train_loss_dict)

    def train(self, epochs=100):

        # load data
        train_loader, test_loader = self.get_train_test_loader()

        # define model
        model = self.get_model()

        # init training parameter and wandb
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        model_path = self.get_model_path()

        # init wandb
        self.init_wandb(model_path, len(train_loader.dataset), len(test_loader.dataset))

        # train loop
        best_test_loss = sys.maxsize
        print("start training")
        for epoch in tqdm(range(epochs)):
            train_loss = self.train_step(model, train_loader, optimizer)
            test_loss = self.test_step(model, test_loader)
            self.log_loss(train_loss, test_loss)
            if test_loss.overal < best_test_loss:
                utils.save_model_checkpoint(model, epoch, model_path)

    def get_ssl_representations(self):
        model = self.load_model()
        model.to(self.device)
        x = self.dataset.adata.X.toarray()
        x = torch.tensor(x, device=self.device)
        latent, _, prot_dist = model.prototype_model(x)
        projections = model.barlow_model.projector(prot_dist)
        return (
            utils.tensor_to_numpy(latent),
            utils.tensor_to_numpy(prot_dist),
            utils.tensor_to_numpy(projections),
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
                print(model_path, " exist")
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

            # Add the fold column to ref and query dataframes
            fold_df["fold"] = self.fold

            # Append the fold dataframes to the overall dataframes
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

    def linear_classification(self):
        _, _, X = self.get_ref_query_latent()
        labels = self.dataset.adata.obs.cell_type.values
        return train_linear_classifier(X, labels)

    def train_custom_cross_val(self, epochs):
        self.custom_cross_val = True
        print("running custom cross validation")
        study_ids = self.dataset.get_study_ids()
        num_studies = len(study_ids)

        # Generate all combinations of test sets with exactly two studies
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

            self.fold += 1
            model_path = self.get_model_path()
            if os.path.exists(model_path):
                print(model_path, " exist")
                continue

            self.train(epochs)

    def evaluate_custom_cross_val_models(self, evaluation_fns):
        print(f"running custom cross validation evaluation for {self.get_model_name()}")
        self.custom_cross_val = True
        study_ids = self.dataset.get_study_ids()
        num_studies = len(study_ids)

        # Generate all combinations of test sets with exactly two studies
        test_combinations = list(combinations(range(num_studies), 2))

        self.set_fold(0)
        overall_df_list = [pd.DataFrame() for evaluation_fn in evaluation_fns]

        for test_study_indices in test_combinations:
            train_study_indices = [
                i for i in range(num_studies) if i not in test_study_indices
            ]
            test_study_indices = list(test_study_indices)
            self.ref, self.query = self.dataset.get_fold_train_test(
                train_study_indices, test_study_indices
            )

            self.set_fold(self.fold + 1)
            print(f"Evaluating fold {self.fold}")

            fold_df_list = [evaluation_fn() for evaluation_fn in evaluation_fns]

            for i, _ in enumerate(evaluation_fns):
                # Add the fold column to the evaluation dataframe
                fold_df_list[i]["fold"] = self.fold
                fold_df_list[i]['trainer'] = self.get_model_name()
                # Append the fold dataframes to the overall dataframe
                overall_df_list[i] = overall_df_list[i].append(
                    fold_df_list[i], ignore_index=True
                )

        return overall_df_list
    

    def get_query_model_path(self):
        model_path = self.get_model_path()
        base, ext = os.path.splitext(model_path)
        query_suffix = "query"
        if self.fine_tuning_epochs:
            query_suffix += f"_e{self.fine_tuning_epochs}"

        if ext:
            new_file_path = f"{base}_{query_suffix}{ext}"
        else:
            new_file_path = f"{model_path}_{query_suffix}"

        return new_file_path

    def get_query_model(self):
        pass

    def finetune_query_model(self, model):
        pass

    def load_query_model(self, model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model 
    
    def load_or_finetune_query_model(self):
        model = self.get_query_model()
        if not self.fine_tuning_epochs:
            return model
        path = self.get_query_model_path()
        if os.path.exists(path):
            return self.load_query_model(model, path)
        else:
            return self.finetune_query_model(model)

    def get_query_model_latent(self, model, adata):
        pass

    def calculate_ref_query_latent(self):
        # Setup AnnData for scVI using the same settings as the reference data
        model = self.load_or_finetune_query_model()
        query_latent = self.get_query_model_latent(model, self.query.adata)
        ref_latent = self.get_query_model_latent(model, self.ref.adata)
        all_latent = self.get_query_model_latent(model, self.dataset.adata)
        return ref_latent, query_latent, all_latent
