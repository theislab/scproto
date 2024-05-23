# import interpretable_ssl.pbmc.label_encoder as pbmc_label_encoder
# import interpretable_ssl.pbmc.train as pbmc_train
# import interpretable_ssl.pancras.train.train as pancras_train
import interpretable_ssl.immune.trainer as immune_trainer
import interpretable_ssl.pbmc3k.trainer as pbmc3k_trainer

from interpretable_ssl.trainers.scpoli_trainer import *
from interpretable_ssl.trainers.scpoli_cvae import CvaeTrainer
# def pancreas():
#     trainer = pancras_train.PancrasTrainer(split_study=True)
#     trainer.train()
    
def immune_ssl():
    trainer = immune_trainer.ImmuneTrainer(self_supervised=True)
    trainer.train()
    
def pbmc3k():
    trainer = pbmc3k_trainer.PBMC3kClassifierTrainer()
    trainer.train()

def pbmc3k_ssl():
    trainer = pbmc3k_trainer.PBMC3kSSLTrainer()
    trainer.train()
    
def prototype_scpoli():
    ScpoliTrainer().train(100)
   
def scpoli_cvae():
    print('train only scpoli cvae')
    CvaeTrainer().train(100) 
    
def linear_scpoli():
    print('training linear classifier with scpoli cvae, , task ratio=10')
    LinearTrainer().train(100)
    
def barlow_scpoli():
    print('running barlow with scpoli cvae as encoder')
    BarlowTrainer().train(100)
if __name__ == "__main__":
    # barlow_scpoli()
    # linear_scpoli()
    # ScpoliOriginal().train()
    prototype_scpoli()