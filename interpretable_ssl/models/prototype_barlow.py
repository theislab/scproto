from barlow_twins_pytorch.Twins.barlow import *
from interpretable_ssl.models.autoencoder import VariationalAutoencoder
from interpretable_ssl.models.prototype_model import PrototypeModel, PrototypeLoss
from interpretable_ssl.pbmc3k.dataset import PBMC3kDataset
import torch
import torch
import torch.nn as nn


class PrototypeBarlow(nn.Module):
    def __init__(self, vae: VariationalAutoencoder, num_prototypes) -> None:
        super(PrototypeBarlow, self).__init__()
        self.prototype_model = PrototypeModel(vae, num_prototypes)
        projection_sizes, lambd = [num_prototypes, 32, 32, 32], 3.9e-3

        self.barlow_model = BarlowProjector(projection_sizes, lambd)

    def forward(self, x1, x2):
        z1, x_hat1, prot_dist1 = self.prototype_model(x1)
        z2, x_hat2, prot_dist2 = self.prototype_model(x2)
        barlow_loss = self.barlow_model(prot_dist1, prot_dist2)

        prot_loss1 = PrototypeLoss()
        prot_loss1.calculate(x1, x_hat1, z1, self.prototype_model)

        prot_loss2 = PrototypeLoss()
        prot_loss2.calculate(x2, x_hat2, z2, self.prototype_model)

        batch_loss = prot_loss1 + prot_loss2
        batch_loss.set_model_loss(barlow_loss)
        return batch_loss


class BarlowProjector(nn.Module):
    def __init__(self, projection_sizes, lambd, scale_factor=1) -> None:
        super(BarlowProjector, self).__init__()
        self.lambd = lambd
        self.scale_factor = scale_factor
        # projector
        sizes = projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, zp1, zp2):
        z1 = self.projector(zp1)
        z2 = self.projector(zp2)

        # empirical cross-correlation matrix
        c = torch.mm(self.bn(z1).T, self.bn(z2))
        c.div_(z1.shape[0])

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)
        return loss

# moved to train_utils

# def optimize_model(x1, x2, model, optimizer, overal_loss):

    
# def train_step(model: PrototypeBarlow, data_loader, optimizer, device):



# def test_step(model, data_loader, device):
