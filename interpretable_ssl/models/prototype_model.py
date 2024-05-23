import torch
import torch.nn as nn
import wandb
from interpretable_ssl.models.autoencoder import vae_loss, PrototypeVAE
from torcheval.metrics.functional import multiclass_f1_score

# from torchvision import datasets, transforms
import torch.optim as optim
import time
from tqdm.auto import tqdm
import interpretable_ssl.utils as utils


class PrototypeBase(nn.Module):
    def __init__(self, num_prototypes, latent_dims) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_shape = (self.num_prototypes, latent_dims)
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

    def forward(self, z):
        p_dist = self.prototype_distance(z)
        f_dist = self.feature_vector_distance(z)
        return (
            self.reg1 * p_dist.min(1).values.mean()
            + self.reg2 * f_dist.min(1).values.mean()
        )


class PrototypeModel(PrototypeBase):
    def __init__(self, vae: PrototypeVAE, num_prototypes) -> None:
        super(PrototypeModel, self).__init__(
            num_prototypes=num_prototypes, latent_dims=self.vae.get_latent_dims()
        )
        self.vae = vae

    def prototype_forward(self, x):
        z = self.vae.encode(x)
        p_dist = self.prototype_distance(z)
        p_dist = p_dist.reshape(-1, self.num_prototypes)
        return z, p_dist

    def forward(self, x):
        z, p_dist = self.prototype_forward(x)
        # output = self.model(p_dist)
        # y = torch.softmax(logits, dim=1)
        return z, self.vae.decode(z), p_dist


class PrototypeLoss:
    def __init__(self) -> None:
        # print('------0.01 vae_reg---------')
        self.vae, self.interpretability = 0, 0
        self.overal = 0
        self.task = 0
        self.vae_reg = 0.01
        self.fixed_values = ['vae_reg', 'fixed_values']
        # self.to_norm_loss_keys = ['vae', 'task', 'interpretability']
        # self.max_values = {self.__dict__[key] for key in self.to_norm_loss_keys}
    
    def calculate_overal(self, vae, interpretability, task=0):
        self.vae = vae
        self.interpretability = interpretability
        self.task = task
        self.overal = self.interpretability + self.vae_reg * self.vae + self.task
        # self.overal = self.vae

    def __add__(self, prot_loss):
        new_loss = PrototypeLoss()
        for key in self.__dict__:
            if key in self.fixed_values:
                continue
            new_loss.__dict__[key] = prot_loss.__dict__[key] + self.__dict__[key]
        return new_loss

    def normalize(self, data_loader_size):
        for key, val in self.__dict__.items():
            if key in self.fixed_values:
                continue
            self.__dict__[key] = val / data_loader_size
            
        return self

    def set_task_loss(self, task_loss, task_ratio=1):
        self.task = task_loss
        self.overal += self.task * task_ratio