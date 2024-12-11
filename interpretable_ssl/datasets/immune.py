from interpretable_ssl.datasets.dataset import SingleCellDataset
from pathlib import Path
import pickle as pkl
from copy import deepcopy


def get_label_encoder_path():
    return "./data/pbmc_immune_label_encoder.pkl"


class ImmuneDataset(SingleCellDataset):
    def __init__(self, adata=None, original_idx=None):
        super().__init__("pbmc-immune", adata, get_label_encoder_path(), original_idx)

    def get_data_path(self):
        return Path.home() / "data/scpoli/pbmc_marker.h5ad"

    def get_test_studies(self):
        return ["Freytag", "Villani"]
