from interpretable_ssl.dataset import SingleCellDataset
from pathlib import Path
import pickle as pkl
from copy import deepcopy
def get_label_encoder_path():
    return "./data/pbmc_immune_label_encoder.pkl"
    
class ImmuneDataset(SingleCellDataset):
    def __init__(self, adata=None, use_pca=False, self_supervised=False):
        super().__init__('pbmc-immune', adata, use_pca, self_supervised)
    
    def get_data_path(self):
        return Path.home() / "data/scpoli/pbmc-immune-processed.h5ad"
    
    def load_label_encoder(self):
        path = get_label_encoder_path()
        return pkl.load(open(path, 'rb'))
    
    def get_train_test(self):
        test_studies = ['Freytag', 'Villani']
        test_idx = self.adata.obs.study.isin(test_studies)
        train, test = self.adata[~test_idx], self.adata[test_idx]
        train_ds, test_ds = deepcopy(self), deepcopy(self)
        train_ds.set_adata(train)
        test_ds.set_adata(test)
        return train_ds, test_ds