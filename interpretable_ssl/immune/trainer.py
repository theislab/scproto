from interpretable_ssl.trainers.trainer import Trainer
import interpretable_ssl.utils as utils
from interpretable_ssl.immune.dataset import ImmuneDataset
from pathlib import Path

class ImmuneTrainer(Trainer):
    def __init__(self, partially_train_ratio=0.5, self_supervised=False, split_study=True) -> None:
        super().__init__(partially_train_ratio, self_supervised)
        self.split_study = split_study
        self.batch_size = 256
        
        self.latent_dims = 8
        self.hidden_dim = 32
        self.num_prototypes = 32
        self.epochs = 300
        print(f'training with number of prototypes : {self.get_model_name()}')
        
    def get_dataset(self):
        return ImmuneDataset()