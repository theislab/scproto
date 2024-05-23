# load data
# define model
# set training parameters
# train loop

from interpretable_ssl.immune.dataset import ImmuneDataset
from scarches.models.scpoli import scPoli
from interpretable_ssl import utils
from interpretable_ssl.trainers.scpoli_trainer import ScpoliTrainer
import torch
import wandb
import sys
class CvaeTrainer(ScpoliTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.experiment_name = "scpoli_cvae"

    def get_model(self, adata):
        condition_key = "study"
        cell_type_key = "cell_type"
        return scPoli(
            adata=adata,
            condition_keys=[condition_key],
            cell_type_keys=[cell_type_key],
            latent_dim=self.latent_dims,
            recon_loss="nb",
        )

    def train_step(self, train_loader, optimizer, model):
        overal_loss = 0
        for scpoli_batch in train_loader:
            scpoli_batch = {key: scpoli_batch[key].to(self.device) for key in scpoli_batch}
            optimizer.zero_grad()

            z, recon_loss, kl_loss, mmd_loss = model(**scpoli_batch)
            cvae_loss = recon_loss + 0.5 * kl_loss + mmd_loss
            cvae_loss.backward()

            optimizer.step()

            overal_loss += cvae_loss
        return overal_loss / len(train_loader)

    def test_step(self, data_loader, model):

        test_loss = 0
        model.eval()  # put model in eval mode

        # Turn on inference context manager
        with torch.inference_mode():
            for scpoli_batch in data_loader:
                # x1, x2 = x1.to(device), x2.to(device)

                # 1. Forward pass
                # 2. Calculate loss
                scpoli_batch = {key: scpoli_batch[key].to(self.device) for key in scpoli_batch}
                z, recon_loss, kl_loss, mmd_loss = model(**scpoli_batch)
                cvae_loss = recon_loss + 0.5 * kl_loss + mmd_loss
                test_loss += cvae_loss

        return test_loss / len(data_loader)
    def get_model_name(self):
        return f'scpoli-cvae-latent_dim{self.latent_dims}.pth'
    
    def train(self, epochs):
        ref, query = self.dataset.get_train_test()

        model = self.get_model(ref.adata).model
        model.to(self.device)

        train_adata, train_loader, val_adata, val_loader = (
            self.prepare_scpoli_data_splits(ref, model)
        )

        # init training parameter and wandb
        optimizer = self.get_optimizer(model)
        model_path = self.get_model_path()

        self.init_wandb(model_path, len(ref), len(query))
        best_test_loss = sys.maxsize
        for epoch in range(epochs):
            train_loss = self.train_step(train_loader, optimizer, model)
            test_loss = self.test_step(val_loader, model)
            wandb.log({'train loss': train_loss, 'test loss': test_loss})
            if test_loss < best_test_loss:
                utils.save_model_checkpoint(model, optimizer, epoch, model_path)
  
            