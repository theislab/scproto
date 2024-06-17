# import interpretable_ssl.pbmc.label_encoder as pbmc_label_encoder
# import interpretable_ssl.pbmc.train as pbmc_train
# import interpretable_ssl.pancras.train.train as pancras_train

from interpretable_ssl.trainers.scpoli_trainer import *
from interpretable_ssl.trainers.scpoli_original import OriginalTrainer
# def pancreas():
#     trainer = pancras_train.PancrasTrainer(split_study=True)
#     trainer.train()
from interpretable_ssl.experiments.change_prototype_count import *
from interpretable_ssl.experiments.hlca.scpoli_prot import *
from interpretable_ssl.experiments.immune.scpoli_prot import *
from interpretable_ssl.experiments.immune.original_scpoli import *
from interpretable_ssl.experiments.immune.scvi import *
from interpretable_ssl.experiments.immune.simclr import *

if __name__ == "__main__":
    print('-----main started----')
    # barlow_scpoli()
    # linear_scpoli()
    # ScpoliOriginal().train()
    
    # train_using_diffrent_prototypes()
    # train_hlca_scpoli_prot_barlow()
    # 
    # train_scpoli_using_original_code()
    # scpoli_original_five_fold()
    # scvi_five_fold()
    # scpoli_prot_barlow_immune_five_fold()
    # train_scvi_immune()
    # scpoli_prot_barlow_immune_five_fold()
    # scpoli_prot_barlow_immune_five_fold_weighted()
    # scpoli_prot_barlow_immune_five_fold_prot_16_weighted()
    # scpoli_prot_barlow_immune_five_fold_prot_20_weighted()
    
    # scpoli_prot_barlow_immune_five_fold()
    # scpoli_prot_barlow_immune_five_fold_weighted()
    # scpoli_prot_barlow_immune_five_fold_prot_32()
    # scpoli_prot_barlow_immune_five_fold_prot_32_weighted()
    
    # no five fold
    # train_scpoli_prot_barlow_immune()
    # train_scpoli_prot_barlow_immune_weighted()
    # train_scpoli_prot_barlow_immune_32()
    # train_scvi_immune()
    # train_scpoli_using_original_code()
    
    # scvi_ccross_val_immune()
    # barlow_ccross_val_immune()
    # scpoli_ccross_val_immune()
    
    # barlow_ccross_val_immune(32)
    # barlow_immune_recheck()
    
    # scvi_immune_bs256()
    # scpoli_immune_bs256()
    # barlow_immune_bs256()
    
    # train_simclr_immune()
    # scvi_immune_bs(512)
    # barlow_immune_bs(512)
    # scpoli_immune_bs(512)
    
    # train_hlca_scpoli_prot_barlow_bs(512)
    
    # scvi_ccross_val_immune_bs(512)
    # barlow_ccross_val_immune_bs(512)
    # scpoli_ccross_val_immune_bs(512)
    # simclr_ccross_val_immune_bs(512)
    
    # barlow_ccross_val_immune_bs_prot(512, 32)
    # barlow_ccross_val_immune_bs(1024)
    # barlow_ccross_val_immune_repulsion_bs(1024)
    
    # simclr_ccross_val_immune_bs(1024)
    
    # barlow_prot_epoch(32, 500)
    # simclr_prot_epoch(32, 500)
    # scpoli_epoch(500)
    
    # barlow_pversion1_epoch(100)
    
    # barlow_pversion1_epoch_bs(1000, 1024)
    # scpoli_epoch_bs(1000, 1024)
    
    # barlow_prot_epoch(32, 500)
    # barlow_prot_epoch_bs(32, 500, 1024)
    barlow_prot_epoch_bs_repulsion(32, 500, 1024)
    
    