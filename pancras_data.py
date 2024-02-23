from pathlib import Path
import scanpy as sc
from anndata.experimental.pytorch import AnnLoader
from sklearn.model_selection import train_test_split


def get_data_path():
    return Path.home() / "data/scpoli/pancreas_sparse.h5ad"


def get_model_path():
    return Path.home() / "models/simple-autoencoder.pth"

def load_data(device, batch_size):
    data_path = get_data_path()

    print("loading data")
    data = sc.read_h5ad(data_path)

    train_idx, test_idx = train_test_split(range(len(data)))
    train, test = data[train_idx], data[test_idx]

    x_dim = train[0].shape[1]
    print(f"x_dim : {x_dim}")

    train_loader = AnnLoader(
        train, batch_size=batch_size, shuffle=True, use_cuda=device
    )
    test_loader = AnnLoader(test, batch_size=batch_size, use_cuda=device)
    return train_loader, test_loader, x_dim