from interpretable_ssl.trainers.scvi_trainer import *
from interpretable_ssl.datasets.immune import *


def train_scvi_immune():
    ds = ImmuneDataset()
    trainer = ScviTrainer(dataset = ds)
    trainer.train(100)
    
def scvi_five_fold():
    print('running scpoli original 5 fold')
    ds = ImmuneDataset()
    trainer = ScviTrainer(dataset = ds)
    trainer.train_kfold_cross_val(50)