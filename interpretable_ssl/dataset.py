import scanpy as sc
from torch.utils.data import Dataset
import torch
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import interpretable_ssl.utils as utils
import random

class SingleCellDataset(Dataset):

    def __init__(self, name, adata=None, use_pca=False, self_supervised=False, multiple_augment_cnt=None):
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
        self.self_supervised = self_supervised
        self.multiple_augment_cnt = multiple_augment_cnt

    def set_adata(self, adata):
        self.adata = adata
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
        if self.self_supervised:
            return self.get_self_supervised_item(idx)
        else:
            x = self.get_x(idx).squeeze(0)
            y = self.get_y(idx).squeeze(0)
            return x, y

    def get_self_supervised_item(self, idx):
        x1 = self.get_x(idx).squeeze(0)
        cell_type = self.adata[idx].obs.cell_type
        if self.multiple_augment_cnt:
            x2 = [self.augment(cell_type).squeeze(0) for _ in range(self.multiple_augment_cnt)]
            x2 = torch.stack(x2)
            x2 = x2.to(self.device)
        else:
            x2 = self.augment(cell_type).squeeze(0)
        return x1, x2
    
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

    def get_x(self):
        x = self.adata.X.toarray()
        return torch.tensor(x, device=self.device)
    
    def get_y(self, i):
        y = self.le.transform(self.adata[i].obs["cell_type"])
        return torch.tensor(y, device=self.device)

    def augment(self, cell_type):
        if len(cell_type) > 1:
            res = [self.augment(cell) for cell in cell_type]
            return torch.tensor(res, device = self.device)
        if type(cell_type) != str:
            
            cell_type = cell_type.iloc[0]
        adata = self.adata
        all_cells = adata[adata.obs.cell_type == cell_type]
        rand_idx = random.randint(0, len(all_cells) - 1)
        x = all_cells[rand_idx].X.toarray()
        return torch.tensor(x, device = self.device)
        