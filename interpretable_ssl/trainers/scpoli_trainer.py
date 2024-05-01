from interpretable_ssl.trainers.trainer import Trainer
from interpretable_ssl.models.scpoli import *
from sklearn.model_selection import train_test_split
from interpretable_ssl.immune.dataset import ImmuneDataset

class ScpoliTrainer(Trainer):
    def __init__(self) -> None:
        super().__init__()
        self.latent_dims = 8
        self.num_prototypes = 16
        self.experiment_name = 'scpoli'
        
    def split_train_test(self, ref):
        train_idx, val_idx = train_test_split(range(len(ref.adata)))
        train, val = ref.adata[train_idx], ref.adata[val_idx]
        return train, val
    
    
    def get_dataset(self):
        return ImmuneDataset()

    def train(self, epochs):
        # prepare data (train, test)
        # define model
        # init training params + wandb
        # train loop:
        #   for each epoch:
        #      calculate loss
        #      optimize
        #      log loss
        
        ref, query = self.dataset.get_train_test()        
        model = PrototypeScpoli(ref.adata, self.latent_dims, self.num_prototypes)
        model.to(self.device)
        
        train_adata, val_adata = self.split_train_test(ref)
        train_loader = generate_scpoli_dataloder(train_adata, model.scpoli.model)
        val_loader = generate_scpoli_dataloder(val_adata, model.scpoli.model)
                
        # init training parameter and wandb
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        model_path = self.get_model_path()

        # init wandb
        self.init_wandb(model_path, len(ref), len(query))
        
        for _ in range(epochs):
            train_loss = train_step(model, train_adata, train_loader, optimizer)
            val_loss = test_step(model, val_loader, val_adata)
            self.log_loss(train_loss, val_loss)
        utils.save_model_checkpoint(model, optimizer, epochs, model_path)
