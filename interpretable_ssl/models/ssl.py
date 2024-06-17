from barlow_twins_pytorch.Twins.barlow import *
import torch
import torch
import torch.nn as nn
from interpretable_ssl.models.loss import *

class Projector(nn.Module):
    def __init__(self, projection_sizes, lambd, scale_factor=1) -> None:
        super(Projector, self).__init__()
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
        
        
class BarlowProjector(Projector):
    def __init__(self, projection_sizes, lambd, scale_factor=1) -> None:
        super().__init__(projection_sizes, lambd, scale_factor)
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(projection_sizes[-1], affine=False)
        
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
    
class SimClr(Projector):
    def forward(self, zp1, zp2):
        z1 = self.projector(zp1)
        z2 = self.projector(zp2)
        return nt_xent_loss(z1, z2)
