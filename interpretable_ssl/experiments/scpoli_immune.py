from interpretable_ssl.trainers.trainer import Trainer
import interpretable_ssl.utils as utils
from interpretable_ssl.datasets.immune import ImmuneDataset
from pathlib import Path
from interpretable_ssl.trainers.scpoli_trainer import *

def train_scpoli_prot_barlow_immune():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.experiment_name = 'scpoli-kmean-prot'
    trainer.num_prototypes = 32
    trainer.train(50)
    
def train_scpoli_prot_barlow_immune_weighted_sampling():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.use_weighted_sampling = True
    trainer.experiment_name = 'scpoli-wsampling-100epoch'
    trainer.num_prototypes = 64
    trainer.train(100)