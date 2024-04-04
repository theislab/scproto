import interpretable_ssl.pbmc.label_encoder as pbmc_label_encoder
import interpretable_ssl.pbmc.train as pbmc_train
import interpretable_ssl.pancras.train.train as pancras_train
import interpretable_ssl.immune.trainer as immune_trainer
import interpretable_ssl.pbmc3k.trainer as pbmc3k_trainer
def pancreas():
    trainer = pancras_train.PancrasTrainer(split_study=True)
    trainer.train()
    
def immune():
    trainer = immune_trainer.ImmuneTrainer()
    trainer.train()
    
def pbmc3k():
    trainer = pbmc3k_trainer.PBMC3kTrainer()
    trainer.train()

if __name__ == "__main__":
    pbmc3k()