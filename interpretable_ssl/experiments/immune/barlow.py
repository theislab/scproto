from interpretable_ssl.datasets.immune import ImmuneDataset
from interpretable_ssl.trainers.scpoli_trainer import *

def get_barlow_immune_trainer():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    return trainer

def train_scpoli_prot_barlow_immune():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.experiment_name = 'barlow'
    trainer.num_prototypes = 16
    trainer.train(100)
    
def train_scpoli_prot_barlow_immune_weighted():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.experiment_name = 'barlow-weighted'
    trainer.use_weighted_sampling = True
    trainer.num_prototypes = 16
    trainer.train(100)
    
def train_scpoli_prot_barlow_immune_32():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.experiment_name = 'barlow'
    trainer.num_prototypes = 32
    trainer.train(100)
    
def scpoli_prot_barlow_immune_five_fold():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.experiment_name = 'barlow'
    trainer.num_prototypes = 16
    trainer.train_kfold_cross_val(50)
    
def scpoli_prot_barlow_immune_five_fold_weighted():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.use_weighted_sampling = True
    trainer.experiment_name = 'barlow-weighted'
    trainer.num_prototypes = 16
    trainer.train_kfold_cross_val(50)
    
    
def scpoli_prot_barlow_immune_five_fold_prot_32():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.experiment_name = 'barlow'
    trainer.num_prototypes = 32
    trainer.train_kfold_cross_val(50)
    
def scpoli_prot_barlow_immune_five_fold_prot_32_weighted():
    dataset = ImmuneDataset()
    trainer = ScpoliProtBarlowTrainer(dataset)
    trainer.use_weighted_sampling = True
    trainer.experiment_name = 'barlow-weighted'
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
    
def barlow_ccross_val_immune(num_prototypes = 16):
    trainer = get_barlow_immune_trainer()
    trainer.num_prototypes = num_prototypes
    trainer.train_custom_cross_val(100)
    
def barlow_immune_recheck():
    trainer = get_barlow_immune_trainer()
    trainer.experiment_name = 'barlow-recheck'
    trainer.train(100)
    
def barlow_immune_bs256():
    trainer = get_barlow_immune_trainer()
    trainer.batch_size = 256
    trainer.train(50)
    
    
def barlow_immune_bs(bs):
    trainer = get_barlow_immune_trainer()
    trainer.batch_size = bs
    trainer.train(100)
    
def barlow_ccross_val_immune_bs(bs):
    trainer = get_barlow_immune_trainer()
    trainer.batch_size = bs
    trainer.train_custom_cross_val(100)
    
def barlow_ccross_val_immune_bs_prot(bs, num_prototypes):
    trainer = get_barlow_immune_trainer()
    trainer.batch_size = bs
    trainer.num_prototypes = num_prototypes
    trainer.train_custom_cross_val(100)
    
def barlow_ccross_val_immune_repulsion_bs(bs):
    trainer = get_barlow_immune_trainer()
    trainer.batch_size = bs
    trainer.experiment_name += 'repulsion'
    trainer.train_custom_cross_val(100)

def barlow_prot_epoch(prot_cnt, epochs):
    trainer = get_barlow_immune_trainer()
    trainer.num_prototypes = prot_cnt
    trainer.experiment_name += f'_e{epochs}'
    trainer.train(epochs)

def barlow_pversion1_epoch(epochs):
    trainer = get_barlow_immune_trainer()
    trainer.num_prototypes = 32
    trainer.projection_version = 1
    trainer.experiment_name += f'_e{epochs}'
    trainer.train(epochs)
    
def barlow_pversion1_epoch_bs(epochs, bs):
    trainer = get_barlow_immune_trainer()
    trainer.num_prototypes = 32
    trainer.projection_version = 1
    trainer.batch_size = bs
    trainer.experiment_name += f'_e{epochs}'
    trainer.train(epochs)
    
def barlow_prot_epoch_bs(prot_cnt, epochs, bs):
    trainer = get_barlow_immune_trainer()
    trainer.num_prototypes = prot_cnt
    trainer.batch_size = bs
    trainer.experiment_name += f'_e{epochs}'
    trainer.train(epochs)
    
def barlow_prot_epoch_bs_repulsion(prot_cnt, epochs, bs):
    trainer = get_barlow_immune_trainer()
    trainer.num_prototypes = prot_cnt
    trainer.batch_size = bs
    trainer.experiment_name += f'_e{epochs}_repulsion'
    trainer.train(epochs)