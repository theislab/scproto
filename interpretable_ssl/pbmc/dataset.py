from interpretable_ssl.datasets.dataset import SingleCellDataset
from pathlib import Path
from interpretable_ssl.pbmc.label_encoder import load_label_encoder

class PBMCDataset(SingleCellDataset):
    def __init__(self, use_pca=False):
        
        super().__init__('pbmc', use_pca=use_pca)
    
    def get_data_path(self):
        return Path.home() / "data/scpoli/pbmc_raw.h5ad"
    
    def load_label_encoder(self):
        return load_label_encoder()
        