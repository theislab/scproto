import interpretable_ssl.pancras.dataset as dataset
import pickle as pkl
import interpretable_ssl.utils as utils


def get_label_encoder_path():
    return "./data/pancras_label_encoder.pkl"


def load_label_encoder():
    return pkl.load(open(get_label_encoder_path(), "rb"))


def main():
    # load pancras data
    adata = dataset.PancrasDataset().adata

    utils.fit_label_encoder(adata, get_label_encoder_path())
