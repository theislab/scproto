from interpretable_ssl.trainers.classifier_trainer import ClassifierTrainer
from interpretable_ssl.trainers.ssl_trainer import SSlTrainer

import interpretable_ssl.utils as utils
from interpretable_ssl.pbmc3k.dataset import PBMC3kDataset
from pathlib import Path

class PBMC3kClassifierTrainer(ClassifierTrainer):
    def __init__(self, partially_train_ratio=None, split_study=False) -> None:
        self.split_study = split_study
        super().__init__(partially_train_ratio)
        self.batch_size = 128
        
        # # self.latent_dims = 16
        self.hidden_dim = 64
        self.num_prototypes = 32
        print(f'model name : {self.get_model_name()}')
        
    def get_dataset(self):
        return PBMC3kDataset()
    
class PBMC3kSSLTrainer(SSlTrainer):
    def __init__(self, partially_train_ratio=None) -> None:
        super().__init__(partially_train_ratio)
        self.batch_size = 128
        
        # # self.latent_dims = 16
        self.hidden_dim = 64
        self.num_prototypes = 16
        print(f'model name : {self.get_model_name()}')
        
    def get_dataset(self):
        return PBMC3kDataset(self_supervised=True)
