from interpretable_ssl.datasets.hlca import HLCADataset
from interpretable_ssl.trainers.scpoli_original import *

def get_scpoli_hlca_trainer():
    ds = HLCADataset()
    trainer = OriginalTrainer(ds)
    return trainer

def train_cv_scpoli_hlca_twice_label_prot():
    trainer = get_scpoli_hlca_trainer()
    trainer.batch_size = 512
    trainer.num_prototypes = trainer.num_classes * 2
    trainer.train_custom_cross_val(100)