from interpretable_ssl.vae_protc_trainer import Trainer
from interpretable_ssl.pancras.dataset import *
import interpretable_ssl.utils as utils

class PancrasTrainer(Trainer):
    def __init__(self, partially_train_ratio=None, split_study=False) -> None:
        self.split_study = split_study
        super().__init__(partially_train_ratio)
        self.batch_size = 32
        self.hidden_dim = 16
        self.num_prototypes = 16
        
    def get_dataset(self):
        return PancrasDataset(split_study=self.split_study)
    
    def get_model_path(self):
        name = self.get_model_name()
        if self.split_study:
            name = 'split-study_' + name
        return utils.get_model_dir() + '/pancras/' + name
    
def main():
    trainer = PancrasTrainer(split_study=True)
    trainer.train()