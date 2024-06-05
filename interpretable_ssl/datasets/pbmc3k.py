from interpretable_ssl.datasets.dataset import SingleCellDataset
from pathlib import Path
import pickle as pkl

def get_label_encoder_path():
    return "./data/pbmc3k_label_encoder.pkl"
    
class PBMC3kDataset(SingleCellDataset):
    
    def __init__(self, adata=None, use_pca=False, self_supervised=False):
        super().__init__('pbmc3k', adata, use_pca, self_supervised)
        
    def get_data_path(self):
        return Path.home() / "data/pbmc3k_withoutX.h5ad"
    
    def load_label_encoder(self):
        path = get_label_encoder_path()
        return pkl.load(open(path, 'rb'))
    