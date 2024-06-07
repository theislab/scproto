from interpretable_ssl.trainers.scpoli_trainer import *
from interpretable_ssl.datasets.hlca import *

def train_hlca_scpoli_prot_barlow():
    dataset = HLCADataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.train(50)