import torch

from interpretable_ssl.trainers.scpoli_trainer import ScpoliTrainer
import wandb
from tqdm import tqdm
import os

class CvaeTrainer(ScpoliTrainer):
    def __init__(self, model, batch_size, debug=False, dataset=None, ref_query=None, original_ref=None) -> None:
        self.is_swav=0
        self.finetuning=False
        self.original_ref = original_ref
        super().__init__(debug, dataset, ref_query, batch_size=batch_size)
        self.model = model
        self.optimizer = self.build_optimizer()
        self.dataloader = self.build_data(self.ref.adata)

    def log_wandb_loss(self, cvae, propagation):
        if not self.debug:
            wandb.log({"cvae": cvae, "propagation": propagation})
        else:
            print({"cvae": cvae, "propagation": propagation})
            
    def build_data(self, adata):
        return self.prepare_scpoli_dataloader(adata, self.model.scpoli_encoder)

    def build_optimizer(self, lr=1e-3, eps=0.01):
        params_embedding = []
        params = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if "embedding" in name:
                    params_embedding.append(p)
                else:
                    params.append(p)

        return torch.optim.Adam(
            [
                {"params": params_embedding, "weight_decay": 0},
                {"params": params},
            ],
            lr=lr,
            eps=eps,
            weight_decay=0.04,
        )
    
    def get_dump_path(self):
        return os.path.join(super().get_dump_path(), 'cvae_pretrain')
    
    def get_umap_path(self, split):
        return [f'temp-res/cvae_pretrain/{split}.png']
    
    def train(self, num_epochs):
        """
        Trains the model for a specified number of epochs.

        Args:
            num_epochs (int): Number of epochs to train.

        Returns:
            nn.Module: The trained model.
        """
        for epoch in range(num_epochs):
            if epoch % 5 == 0:
                # self.plot_ref_umap(name_postfix=f'e{epoch}', model=self.model)
                self.plot_umap(self.model, self.original_ref.adata, f"cvae-e{epoch}")
                
            self.model.train()  # Set the model to training mode
            running_loss = 0.0
            propagation_loss = 0

            for batch_idx, inputs in tqdm(enumerate(self.dataloader)):
                # Move data to the specified device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)

                # Forward pass
                x, projector_out, prot_mapped, cvae_loss, prot_decoding_loss = (
                    self.model(inputs)
                )
                propagation_loss += self.model.propagation(x)

                # Backward pass and optimization
                self.optimizer.zero_grad()  # Clear gradients
                cvae_loss.backward()  # Backpropagation
                self.optimizer.step()  # Update parameters

                running_loss += cvae_loss.item()

            # Log epoch loss
            avg_loss = running_loss / len(self.dataloader)
            propagation_loss /= len(self.dataloader)
            self.log_wandb_loss(avg_loss, propagation_loss)
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        print("Training complete.")
        self.save_metrics()
        self.plot_umap(self.model, self.original_ref.adata, f"cvae-final")
        return self.model
