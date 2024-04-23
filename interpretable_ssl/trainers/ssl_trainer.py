import interpretable_ssl.utils as utils
import torch.optim as optim
import wandb
from tqdm.auto import tqdm
import interpretable_ssl.models.prototype_classifier as prototype_classifier
from interpretable_ssl.models.prototype_classifier import ProtClassifier
from interpretable_ssl.dataset import SingleCellDataset
from torch.utils.data import DataLoader

from interpretable_ssl.trainers.trainer import Trainer
from interpretable_ssl.models import autoencoder, prototype_barlow
import torch

class SSlTrainer(Trainer):
    def __init__(self, partially_train_ratio=None, multiple_augment_cnt=None) -> None:
        super().__init__(partially_train_ratio, self_supervised=True)
        self.dataset.multiple_augment_cnt = multiple_augment_cnt
    
    def get_model_name(self):
        name = super().get
    def get_model(self):
        vae = autoencoder.VariationalAutoencoder(self.input_dim, self.hidden_dim, self.latent_dims)
        model = prototype_barlow.PrototypeBarlow(vae, self.num_prototypes)
        return model
    
    def train_step(self, model, train_loader, optimizer):
        return prototype_barlow.train_step(model, train_loader, optimizer, self.device)
    
    def test_step(self, model, test_loader):
        return prototype_barlow.test_step(model, test_loader, self.device)
    
    def get_model_name(self):
        name = super().get_model_name()
        return f'ssl-{name}'
    
