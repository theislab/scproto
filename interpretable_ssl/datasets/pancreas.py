from pathlib import Path
from interpretable_ssl.datasets.dataset import SingleCellDataset
from constants import HOME

def get_label_encoder_path():
    return "./data/pancras_label_encoder.pkl"


class PancreasDataset(SingleCellDataset):

    def __init__(self, adata=None, original_idx=None):
        super().__init__("pancreas", adata, get_label_encoder_path(), original_idx)

    def get_data_path(self):
        return f"{HOME}/data/pancreas.h5ad"

    def get_test_studies(self):
        return ["celseq", "celseq2"]
