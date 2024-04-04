from interpretable_ssl.trainer import Trainer
import interpretable_ssl.utils as utils
from interpretable_ssl.pbmc3k.dataset import PBMC3kDataset
from pathlib import Path

class PBMC3kTrainer(Trainer):
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
    
    def get_model_path(self):
        name = self.get_model_name()
        save_dir = utils.get_model_dir() + '/pbmc3k/'
        Path.mkdir(Path(save_dir), exist_ok=True)
        return save_dir + name