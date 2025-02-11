from interpretable_ssl.datasets.dataset import SingleCellDataset
from constants import HOME

def get_label_encoder_path():
    return f"{HOME}/data/hlca_label_encoder.pkl"


class HLCADataset(SingleCellDataset):
    def __init__(self, adata=None, original_idx=None):
        super().__init__("hlca", adata, get_label_encoder_path(), original_idx)
        # self.adata.obs.rename(columns={'study': 'batch'}, inplace=True)
        self.adata.obs["batch"] = self.adata.obs["study"]

    def get_data_path(self):
        return f"{HOME}/data/hlca/hlca_core_hvg.h5ad"

    def get_test_studies(self):
        return ["Teichmann_Meyer_2019", "Lafyatis_Rojas_2019"]
