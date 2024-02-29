from pathlib import Path
import scanpy as sc
from anndata.experimental.pytorch import AnnLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import random_split
from _label_encoder import *


def get_data_path():
    return Path.home() / "data/scpoli/pancreas_sparse-pca.h5ad"


def get_model_path():
    return Path.home() / "models/simple-autoencoder.pth"


def read_adata():
    data_path = get_data_path()

    print("loading data")
    data = sc.read_h5ad(data_path)
    return data


class PancrasDataset(Dataset):

    def __init__(self, device, use_pca=False):
        print("hi")
        self.device = device
        self.adata = read_adata()
        self.le = load_label_encoder()
        self.use_pca = use_pca

        self.x = self.get_x()
        self.y = self.get_y()

    def set_use_pca(self, use_pca):
        self.use_pca = use_pca
        self.x = self.get_x()
        self.y = self.get_y()

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        x = self.x[idx].squeeze(0)
        y = self.y[idx].squeeze(0)
        return x, y

    def get_train_test(self):
        return random_split(
            self, [0.7, 0.3], generator=torch.Generator().manual_seed(42)
        )

    def get_x(self):
        if self.use_pca:
            x = self.adata.obsm["X_pca"]
        else:
            x = self.adata.X.toarray()
        return torch.tensor(x, device=self.device)

    def get_y(self):
        y = self.le.transform(self.adata.obs["cell_type"])
        return torch.tensor(y, device=self.device)
