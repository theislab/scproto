from interpretable_ssl.datasets.immune import ImmuneDataset
from interpretable_ssl.trainers.scpoli_trainer import *

def get_simclr_immune_trainer():
    dataset = ImmuneDataset()
    trainer = SimClrTrainer(dataset)
    return trainer

def train_simclr_immune():
    trainer = get_simclr_immune_trainer()
    trainer.num_prototypes = 16
    trainer.batch_size = 512
    trainer.train(100)
    
def simclr_ccross_val_immune_bs(bs):
    trainer = get_simclr_immune_trainer()
    trainer.batch_size = bs
    trainer.train_custom_cross_val(100)
    
def simclr_prot_epoch(prot_cnt, epochs):
    trainer = get_simclr_immune_trainer()
    trainer.num_prototypes = prot_cnt
    trainer.experiment_name += f'_e{epochs}'
    trainer.train(epochs)