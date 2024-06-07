from interpretable_ssl.datasets.immune import ImmuneDataset
from interpretable_ssl.trainers.scpoli_original import *

def train_scpoli_using_original_code():
    ds = ImmuneDataset()
    trainer = OriginalTrainer(ds)
    trainer.train(100)
    
def scpoli_original_five_fold():
    print('running scpoli original 5 fold')
    ds = ImmuneDataset()
    trainer = OriginalTrainer(ds)
    trainer.train_kfold_cross_val(50)