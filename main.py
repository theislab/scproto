# import interpretable_ssl.pbmc.label_encoder as pbmc_label_encoder
# import interpretable_ssl.pbmc.train as pbmc_train
# import interpretable_ssl.pancras.train.train as pancras_train
import interpretable_ssl.immune.trainer as immune_trainer
import interpretable_ssl.pbmc3k.trainer as pbmc3k_trainer

from interpretable_ssl.trainers.scpoli_trainer import ScpoliTrainer
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
    
if __name__ == "__main__":
    print('add scpoli_model + try to learn everything')
    ScpoliTrainer().train(100)