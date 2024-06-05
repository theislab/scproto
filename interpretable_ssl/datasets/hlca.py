from interpretable_ssl.datasets.dataset import SingleCellDataset
from pathlib import Path
import pickle as pkl
from copy import deepcopy

def get_label_encoder_path():
    return "./data/hlca_label_encoder.pkl"
    
class HLCADataset(SingleCellDataset):
    def __init__(self, adata=None, use_pca=False, self_supervised=False):
        super().__init__('hlca', adata, use_pca, self_supervised, label_encoder_path=get_label_encoder_path())
        self.get_train_test()
    def get_data_path(self):
        return "/home/icb/fatemehs.hashemig/data/hlca/hlca_core_hvg.h5ad"
    
    
    