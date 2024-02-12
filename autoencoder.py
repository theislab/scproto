import torch.nn as nn
import scanpy as sc
import torch
from anndata.experimental.pytorch import AnnLoader
import torch.optim as optim
import wandb
from tqdm.auto import tqdm
from pathlib import Path

import time


# Defining Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, encoding_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_data(device, batch_size, n_top_genes = None):
    data_path = (
        "/lustre/groups/ml01/workspace/mojtaba.bahrami/proformer/data/cxg_10M_subset/"
    )
    filename = "adata_0.h5ad"
    train_path = f"{data_path}/train/{filename}"
    test_path = f"{data_path}/test/{filename}"

    print('loading train data')
    train = sc.read_h5ad(train_path)
    
    print('loading test data')
    test = sc.read_h5ad(test_path)
    
    if n_top_genes:
        train = sc.pp.highly_variable_genes(train, n_top_genes=n_top_genes)
        train = sc.pp.highly_variable_genes(test, n_top_genes = n_top_genes)
        
    x_dim = train[0].shape[1]
    print(f'x_dim : {x_dim}')
    
    train_loader = AnnLoader(
        train, batch_size=batch_size, shuffle=True, use_cuda=device
    )
    test_loader = AnnLoader(test, batch_size=batch_size, use_cuda=device)
    return train_loader, test_loader, x_dim


def train_step(model, data_loader, loss_fn, optimizer, device):
    train_loss = 0
    model.to(device)
    for batch, data in enumerate(data_loader):

        # 1. Forward pass
        y_pred = model(data.X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, data.X)
        train_loss += loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    train_loss /= len(data_loader)
    return train_loss


def test_step(data_loader, model, loss_fn, device):
    test_loss = 0
    model.to(device)
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for data in data_loader:

            # 1. Forward pass
            test_pred = model(data.X)

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, data.X)

        # Adjust metrics and print out
        test_loss /= len(data_loader)
    return test_loss


def save_model_checkpoint(model, opt, epoch, save_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        },
        save_path,
    )


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # load data
    batch_size = 16
    train_loader, test_loader, input_size = load_data(device, batch_size, 256)

    # define model
    encoding_dim = 16
    hidden_dim = 32
    model = Autoencoder(input_size, encoding_dim, hidden_dim)

    # init training parameters
    epochs = 1000
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    model_path = Path.home() / "models/simple-autoencoder.pth"

    # init wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="interpretable-ssl",
        # track hyperparameters and run metadata
        config={
            "model": "autoencoder",
            "encoding dim": encoding_dim,
            "hidden dim": hidden_dim,
            "epochs": epochs,
            "device": device,
            "model path": model_path,
        },
    )

    best_test_loss = 100000
    print("start training")
    st = time.time()
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, train_loader, criterion, optimizer, device)
        test_loss = test_step(test_loader, model, criterion, device)
        

        wandb.log({"train_loss": train_loss, "test_loss": test_loss})
        
        if best_test_loss > test_loss:
            save_model_checkpoint(model, optimizer, epoch, model_path)
            
    print(f"training took {time.time() - st} seconds")
