from interpretable_ssl.datasets.immune import ImmuneDataset
from interpretable_ssl.trainers.scpoli_trainer import *

def train_scpoli_prot_barlow_immune():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.experiment_name = 'scpoli-kmean-prot'
    trainer.num_prototypes = 32
    trainer.train(50)
    
def scpoli_prot_barlow_immune_five_fold_weighted():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.use_weighted_sampling = True
    trainer.experiment_name = 'barlow-wsampling'
    trainer.num_prototypes = 32
    trainer.train_kfold_cross_val(50)
    
def scpoli_prot_barlow_immune_five_fold():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    # trainer.use_weighted_sampling = True
    trainer.experiment_name = 'barlow'
    trainer.num_prototypes = 32
    trainer.train_kfold_cross_val(50)
    
def scpoli_prot_barlow_immune_five_fold_prot_20_weighted():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.use_weighted_sampling = True
    trainer.experiment_name = 'barlow-weighted'
    trainer.num_prototypes = 20
    trainer.train_kfold_cross_val(50)
    
def scpoli_prot_barlow_immune_five_fold_prot_16_weighted():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.use_weighted_sampling = True
    trainer.experiment_name = 'barlow-weighted'
    trainer.num_prototypes = 16
    trainer.train_kfold_cross_val(50)