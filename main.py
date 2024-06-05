# import interpretable_ssl.pbmc.label_encoder as pbmc_label_encoder
# import interpretable_ssl.pbmc.train as pbmc_train
# import interpretable_ssl.pancras.train.train as pancras_train

from interpretable_ssl.trainers.scpoli_trainer import *
from interpretable_ssl.trainers.scpoli_cvae import CvaeTrainer
# def pancreas():
#     trainer = pancras_train.PancrasTrainer(split_study=True)
#     trainer.train()
from interpretable_ssl.experiments.change_prototype_count import *
from interpretable_ssl.experiments.scpoli_hlca import *
from interpretable_ssl.experiments.scpoli_immune import *

    
if __name__ == "__main__":
    # barlow_scpoli()
    # linear_scpoli()
    # ScpoliOriginal().train()
    
    # train_using_diffrent_prototypes()
    # train_hlca_scpoli_prot_barlow()
    # train_scpoli_prot_barlow_immune()
    train_scpoli_prot_barlow_immune_weighted_sampling()