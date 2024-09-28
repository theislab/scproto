from interpretable_ssl.trainers.scpoli_trainer import *
from torch.utils.data import random_split


class AdoptiveTrainer(ScpoliTrainer):
    def __init__(self, parser=None, **kwargs) -> None:
        super().__init__(parser, **kwargs)
        self.finetune_ds = None
        self.original_ref = None
        self.partial_ref = None
        self.finetuning = False
    # split train dataset
    # pretrain model
    # finetune on small portion of the model
    
    def split_train_data(self, finetune_size=0.1):
        self.original_ref = self.ref
        self.partial_ref, self.finetune_ds = self.ref.split(finetune_size)
        
    def finetune(self):
        pass
    
    def get_scib_file_path(self, split):
        base = f'{split}-scib'
        if self.finetuning:
            base = f'{base}-semi-supervised'
        filename = f'{base}.csv'
        return os.path.join(self.get_dump_path(),filename)
    
    def train_semi_supervised(self):
        self.split_train_data()
        self.ref = self.partial_ref
        self.setup()
        self.train()
        self.finetuning = True
        self.ref = self.finetune_ds
        self.finetune()
        # keep finetuning true for scib evaluation on the run        
        
    
