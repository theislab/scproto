from interpretable_ssl.trainers.scpoli_trainer import *
from interpretable_ssl.datasets.hlca import *

def get_hlca_immune_trainer():
    dataset = HLCADataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    return trainer
def train_hlca_scpoli_prot_barlow():
    trainer = get_hlca_immune_trainer()
    trainer.train(100)
    
def train_hlca_scpoli_prot_barlow_bs(bs):
    trainer = get_hlca_immune_trainer()
    trainer.batch_size = bs
    trainer.train(100)