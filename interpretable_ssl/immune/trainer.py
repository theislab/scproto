from interpretable_ssl.vae_protc_trainer import Trainer
import interpretable_ssl.utils as utils
from interpretable_ssl.immune.dataset import ImmuneDataset
from pathlib import Path
class ImmuneTrainer(Trainer):
    def __init__(self, partially_train_ratio=None, split_study=False) -> None:
        self.split_study = split_study
        super().__init__(partially_train_ratio)
        # self.batch_size = 16
        
        # # self.latent_dims = 16
        # self.hidden_dim = 128
        self.num_prototypes = 32
        print(f'training with number of prototypes : {self.num_prototypes}')
        
    def get_dataset(self):
        return ImmuneDataset()
    
    def get_model_path(self):
        name = self.get_model_name()
        if self.split_study:
            name = 'split-study_' + name
        save_dir = utils.get_model_dir() + '/immune/'
        Path.mkdir(Path(save_dir), exist_ok=True)
        return save_dir + name