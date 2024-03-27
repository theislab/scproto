from interpretable_ssl.vae_protc_trainer import Trainer
from interpretable_ssl.pbmc.dataset import PBMCDataset
import interpretable_ssl.utils as utils

class PBMCTrainer(Trainer):
    def get_dataset(self):
        return PBMCDataset()
    def get_model_path(self):
        base = utils.get_model_dir() + f'pbmc/num-prot-{self.num_prototypes}_hidden-{self.hidden_dim}_bs-{self.batch_size}'
        if self.partially_train_ratio:
            return f'{base}_train-ratio-{self.partially_train_ratio}.pth'
        return base + '.pth'
    
def main():
    trainer = PBMCTrainer(partially_train_ratio=0.01)
    trainer.batch_size = 32
    trainer.hidden_dim = 16
    trainer.num_prototypes = 32
    
    trainer.train()