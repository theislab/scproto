import scanpy as sc
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import interpretable_ssl.utils as utils
import random

class SingleCellDataset(Dataset):

    def __init__(self, name, adata=None, use_pca=False):
        self.device = utils.get_device()
        self.name = name
        if not adata:
            self.adata = self.read_adata()
        else:
            self.adata = adata
        self.le = self.load_label_encoder()
        self.use_pca = use_pca
        # self.x = self.get_x()
        # self.y = self.get_y()
        self.num_classes = len(set(self.adata.obs["cell_type"].cat.categories))
        self.x_dim = self.adata[0].X.shape[1]

    def __str__(self) -> str:
        return self.name

    def get_data_path(self):
        pass

    def read_adata(self):
        data_path = self.get_data_path()
        print("loading data")
        data = sc.read_h5ad(data_path)
        return data

    def load_label_encoder(self):
        pass

    # def set_use_pca(self, use_pca):
    #     self.use_pca = use_pca
    #     self.x = self.get_x()
    #     self.y = self.get_y()

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        x = self.get_x(idx).squeeze(0)
        y = self.get_y(idx).squeeze(0)
        return x, y

    def get_train_test(self):
        return random_split(
            self, [0.7, 0.3], generator=torch.Generator().manual_seed(42)
        )

    def get_x(self, i):
        if self.use_pca:
            x = self.adata[i].obsm["X_pca"]
        else:
            x = self.adata[i].X.toarray()
        return torch.tensor(x, device=self.device)

    def get_y(self, i):
        y = self.le.transform(self.adata[i].obs["cell_type"])
        return torch.tensor(y, device=self.device)
