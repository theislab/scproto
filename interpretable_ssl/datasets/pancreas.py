from pathlib import Path
from interpretable_ssl.datasets.dataset import SingleCellDataset


def get_label_encoder_path():
    return "./data/pancras_label_encoder.pkl"


class PancrasDataset(SingleCellDataset):

    def __init__(self, adata=None, original_idx=None):
        super().__init__("pancreas", adata, get_label_encoder_path(), original_idx)

    def get_data_path(self):
        return Path.home() / "data/scpoli/pancreas_sparse-pca.h5ad"

    def get_test_studies(self):
        return ["celseq", "celseq2"]
