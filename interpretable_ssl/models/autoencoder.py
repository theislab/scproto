# https://avandekleut.github.io/vae/
import torch.nn as nn
import torch
import torch.optim as optim
import wandb
from tqdm.auto import tqdm

import time
import torch.nn.functional as F
import interpretable_ssl.pancras.dataset as dataset
import interpretable_ssl.utils as utils

class PrototypeVAE:

    def get_latent_dims(self):
        pass

    def encode(self, x):
        pass

    def decode(self, z):
        pass

    def get_kl(self):
        pass
    
    def calculate_loss(self, x):
        pass
    
class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dims, use_bn=True):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dims)
        self.linear3 = nn.Linear(hidden_dim, latent_dims)
        if use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dim, affine=True)
            self.bn2 = nn.BatchNorm1d(latent_dims, affine=True)
            self.bn3 = nn.BatchNorm1d(latent_dims, affine=True)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn1(self.linear1(x)))
        
        mu = self.linear2(x)
        mu = self.bn2(mu)
        
        sigma = torch.exp(self.bn3(self.linear3(x)))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim, affine=True)
        self.bn2 = nn.BatchNorm1d(input_dim, affine=True)

    def forward(self, z):
        z = F.relu(self.bn1(self.linear1(z)))
        z = torch.sigmoid(self.bn2(self.linear2(z)))
        return z


class VariationalAutoencoder(nn.Module, PrototypeVAE):
    def __init__(self, input_dim, hidden_dim, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim, self.hidden_dim, self.latent_dims = (
            input_dim,
            hidden_dim,
            latent_dims,
        )
        self.encoder = VariationalEncoder(input_dim, hidden_dim, latent_dims)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_latent_dims(self):
        return self.latent_dims

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def get_kl(self):
        return self.encoder.kl

    def calculate_loss(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return vae_loss(x, x_hat, self.encoder.kl)


def vae_loss(x, x_hat, kl):
    return ((x - x_hat) ** 2).sum() + kl

def train_step(model, data_loader, optimizer, device):
    model.to(device)
    overal_loss = 0
    for batch, data in enumerate(data_loader):

        # 1. Forward pass
        x_hat = model(data.X)

        # 2. Calculate loss
        loss = vae_loss(data.X, x_hat, model.encoder.kl)
        overal_loss += loss

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    overal_loss /= len(data_loader)
    return overal_loss


def test_step(data_loader, model, device):
    test_loss = 0
    model.to(device)
    model.eval()  # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for data in data_loader:

            # 1. Forward pass
            x_hat = model(data.X)

            # 2. Calculate loss
            loss = ((data.X - x_hat) ** 2).sum() + model.encoder.kl

            # 2. Calculate loss and accuracy
            test_loss += loss.item()

        # Adjust metrics and print out
        test_loss /= len(data_loader)
    return test_loss





def main():

    device = utils.get_device()
    print(device)

    # load data
    batch_size = 128
    train_loader, test_loader, input_size = dataset.load_data(device, batch_size)

    # define model
    encoding_dim = 128
    hidden_dim = 256
    model = VariationalAutoencoder(input_size, hidden_dim, encoding_dim)

    # init training parameters
    epochs = 1000

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    model_path = dataset.get_model_path()

    # init wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="interpretable-ssl",
        # track hyperparameters and run metadata
        config={
            "model": "vae",
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
        train_loss = train_step(model, train_loader, optimizer, device)
        test_loss = test_step(test_loader, model, device)

        wandb.log({"train_loss": train_loss, "test_loss": test_loss})

        if best_test_loss > test_loss:
            utils.save_model_checkpoint(model, optimizer, epoch, model_path)

    print(f"training took {time.time() - st} seconds")
