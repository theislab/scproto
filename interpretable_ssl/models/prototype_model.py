import torch
import torch.nn as nn
import wandb
from interpretable_ssl.models.autoencoder import vae_loss, VariationalAutoencoder
from torcheval.metrics.functional import multiclass_f1_score

from torchvision import datasets, transforms
import torch.optim as optim
import time
from tqdm.auto import tqdm
import interpretable_ssl.utils as utils


class PrototypeModel(nn.Module):
    def __init__(self, vae: VariationalAutoencoder, num_prototypes) -> None:
        super(PrototypeModel, self).__init__()
        self.vae = vae
        self.num_prototypes = num_prototypes
        self.prototype_shape = (self.num_prototypes, self.vae.latent_dims)
        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )
        self.reg1 = 0.05
        self.reg2 = 0.05
        self.vae_reg = 0.5

    def prototype_distance(self, z: torch.Tensor):
        return torch.cdist(z, self.prototype_vectors)

    def feature_vector_distance(self, z: torch.Tensor):
        return torch.cdist(self.prototype_vectors, z)

    def prototype_forward(self, x):
        z = self.vae.encoder(x)
        p_dist = self.prototype_distance(z)
        p_dist = p_dist.reshape(-1, self.num_prototypes)
        return z, p_dist
    
    def forward(self, x):
        z, p_dist = self.prototype_forward(x)
        # output = self.model(p_dist)
        # y = torch.softmax(logits, dim=1)
        return z, self.vae.decoder(z), p_dist

    def calculate_interpretablity_loss(self, z):
        p_dist = self.prototype_distance(z)
        f_dist = self.feature_vector_distance(z)
        return (
            self.reg1 * p_dist.min(1).values.mean()
            + self.reg2 * f_dist.min(1).values.mean()
        )
    


class PrototypeLoss:
    def __init__(self) -> None:
        self.vae, self.interpretablity = 0, 0
        self.loss = 0
        self.model_loss = 0

    def calculate(self, x, x_hat, z, prot_model: PrototypeModel):
        self.vae = vae_loss(x, x_hat, prot_model.vae.encoder.kl)
        self.interpretablity = prot_model.calculate_interpretablity_loss(z)
        self.loss = (
            self.interpretablity + prot_model.vae_reg * self.vae + self.model_loss
        )

    def __add__(self, prot_loss):
        new_loss = PrototypeLoss()
        for key in self.__dict__:
            new_loss.__dict__[key] = prot_loss.__dict__[key] + self.__dict__[key]
        return new_loss

    def normalize(self, data_loader_size):
        for key, val in self.__dict__.items():
            self.__dict__[key] = val / data_loader_size
        