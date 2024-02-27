from pathlib import Path
import scanpy as sc
from anndata.experimental.pytorch import AnnLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch

def get_data_path():
    return Path.home() / "data/scpoli/pancreas_sparse.h5ad"


def get_model_path():
    return Path.home() / "models/simple-autoencoder.pth"


def read_adata():
    data_path = get_data_path()

    print("loading data")
    data = sc.read_h5ad(data_path)
    return data


class PancrasDataset(Dataset):

    def __init__(self, device):
        print('hi')
        self.device = device
        self.adata = read_adata()
        self.le = LabelEncoder()
        self.le.fit(self.adata.obs["cell_type"])
        
    def __len__(self):
        return len(self.adata)
        
    def __getitem__(self, idx):
        x = self.adata.X[idx].toarray()
        x = torch.tensor(x, device=self.device)
        x = x.squeeze(0)
        
        y = self.adata.obs['cell_type'].iloc[idx]
        y = self.le.transform([y])
        y = torch.tensor(y, device=self.device)
        y = y.squeeze(0)
        return x, y