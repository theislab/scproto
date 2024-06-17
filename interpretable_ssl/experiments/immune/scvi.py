from interpretable_ssl.trainers.scvi_trainer import *
from interpretable_ssl.datasets.immune import *

def get_scvi_immune_trainer():
    ds = ImmuneDataset()
    trainer = ScviTrainer(dataset = ds)
    return trainer

def train_scvi_immune():
    ds = ImmuneDataset()
    trainer = ScviTrainer(dataset = ds)
    trainer.train(100)
    
def scvi_five_fold():
    print('running scpoli original 5 fold')
    ds = ImmuneDataset()
    trainer = ScviTrainer(dataset = ds)
    trainer.train_kfold_cross_val(50)
    
def scvi_ccross_val_immune():
    ds = ImmuneDataset()
    trainer = ScviTrainer(dataset = ds)
    trainer.train_custom_cross_val(100)
    
    
def scvi_immune_bs256():
    trainer = get_scvi_immune_trainer()
    trainer.batch_size = 256
    trainer.train(50)
    
    
def scvi_immune_bs(bs):
    trainer = get_scvi_immune_trainer()
    trainer.batch_size = bs
    trainer.train(100)
    
def scvi_ccross_val_immune_bs(bs):
    ds = ImmuneDataset()
    trainer = ScviTrainer(dataset = ds)
    trainer.batch_size = bs
    trainer.train_custom_cross_val(100)