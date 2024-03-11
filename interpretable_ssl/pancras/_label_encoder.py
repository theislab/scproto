import numpy
from sklearn.preprocessing import LabelEncoder
import interpretable_ssl.pancras.data as data
import pickle as pkl


def get_label_encoder_path():
    return "./data/pancras_label_encoder.pkl"


def load_label_encoder():
    return pkl.load(open(get_label_encoder_path(), 'rb'))


def main():
    # load pancras data
    adata = data.read_adata()

    # fit label encoder
    le = LabelEncoder()
    le.fit(adata.obs["cell_type"])

    # save it
    pkl.dump(le, open(get_label_encoder_path(), 'wb'))
