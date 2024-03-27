import scanpy as sc
import pickle as pkl
import interpretable_ssl.utils as utils


def get_label_encoder_path():
    return "./data/pbmc_label_encoder.pkl"


def load_label_encoder():
    return pkl.load(open(get_label_encoder_path(), "rb"))


def main():
    # load data
    pbmc_path = '/home/icb/fatemehs.hashemig/data/scpoli/pbmc_raw.h5ad'
    adata = sc.read_h5ad(pbmc_path)
    utils.fit_label_encoder(adata, get_label_encoder_path())
