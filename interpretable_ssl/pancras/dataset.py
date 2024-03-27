from pathlib import Path
from interpretable_ssl.pancras.label_encoder import *
from interpretable_ssl.single_cell_dataset import SingleCellDataset



class PancrasDataset(SingleCellDataset):

    def __init__(self, use_pca=False, adata=None, split_study=False):
        self.split_study = split_study
        super().__init__('pancras',adata, use_pca)

    def load_label_encoder(self):
        return load_label_encoder()
    
    def get_data_path(self):
        return Path.home() / "data/scpoli/pancreas_sparse-pca.h5ad"
    
    def get_train_test(self):
        if self.split_study:
            return self.split_train_test_by_study()
        return super().get_train_test()
    
    def split_train_test_by_study(self):
        test_idx = self.adata.obs['study'].isin(['celseq', 'celseq2'])
        train_adata = self.adata[~test_idx]
        test_adata = self.adata[test_idx]
        return PancrasDataset(adata=train_adata), PancrasDataset(adata=test_adata)