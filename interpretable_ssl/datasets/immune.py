from interpretable_ssl.datasets.dataset import SingleCellDataset
from pathlib import Path
from constants import HOME
import scanpy as sc

def get_label_encoder_path():
    return "./data/pbmc_immune_label_encoder.pkl"


class ImmuneDataset(SingleCellDataset):
    def __init__(self, adata=None, original_idx=None):
        super().__init__("pbmc-immune", adata, get_label_encoder_path(), original_idx)

    def get_data_path(self):
        return f"{HOME}/immune.h5ad"

    def get_test_studies(self):
        return ["Freytag", "Villani"]
    
    def preprocess(self):
        sc.pp.highly_variable_genes(self.adata, n_top_genes=4000)
        self.adata = self.adata[:, self.adata.var.highly_variable].copy()  # Keep only HVGs
        self.adata.obs.rename(columns={'final_annotation': 'cell_type', 'tech': 'study'}, inplace=True)
