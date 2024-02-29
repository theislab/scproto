import numpy
from sklearn.preprocessing import LabelEncoder
import pancras_data
import pickle as pkl


def get_label_encoder_path():
    return "./data/pancras_label_encoder.pkl"


def load_label_encoder():
    return pkl.load(open(get_label_encoder_path(), 'rb'))


if __name__ == "__main__":
    # load pancras data
    adata = pancras_data.read_adata()

    # fit label encoder
    le = LabelEncoder()
    le.fit(adata.obs["cell_type"])

    # save it
    pkl.dump(le, open(get_label_encoder_path(), 'wb'))
