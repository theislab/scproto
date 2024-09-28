from interpretable_ssl.datasets.hlca import HLCADataset
from interpretable_ssl.trainers.scpoli_trainer import *

def get_simclr_hlca_trainer():
    dataset = HLCADataset()
    trainer = SimClrTrainer(dataset)
    return trainer

def train_cv_simclr_hlca_twice_label_prot():
    trainer = get_simclr_hlca_trainer()
    trainer.batch_size = 512
    trainer.num_prototypes = trainer.num_classes * 2
    trainer.train_custom_cross_val(100)