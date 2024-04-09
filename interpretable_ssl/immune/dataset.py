from interpretable_ssl.dataset import SingleCellDataset
from pathlib import Path
import pickle as pkl

def get_label_encoder_path():
    return "./data/pbmc_immune_label_encoder.pkl"
    
class ImmuneDataset(SingleCellDataset):
    def __init__(self, use_pca=False, adata=None):
        
        super().__init__('pbmc-immune', use_pca=use_pca, adata=adata)
    
    def get_data_path(self):
        return Path.home() / "data/scpoli/pbmc-immune-processed.h5ad"
    
    def load_label_encoder(self):
        path = get_label_encoder_path()
        return pkl.load(open(path, 'rb'))
    
    def get_train_test(self):
        test_studies = ['Freytag', 'Villani']
        test_idx = self.adata.obs.study.isin(test_studies)
        train, test = self.adata[~test_idx], self.adata[test_idx]
        return ImmuneDataset(adata = train), ImmuneDataset(adata = test)