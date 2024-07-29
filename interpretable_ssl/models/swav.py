import torch.nn as nn
import torch

# encoder
# possibly a projection head
# prototype layer


class SwavModel(nn.Module):
    def __init__(self, scpoli_model, latent_dim, nmb_prototypes):
        super().__init__()
        self.scpoli_model = scpoli_model
        self.prototypes = nn.Linear(latent_dim, nmb_prototypes, bias=False)
        self.projection_head = None
        self.l2norm = True
        print("swav model with l2n init")
        self.reg1 = 0.5
        self.reg2 = 0.5

    def forward(self, batch):
        x, recon_loss, kl_loss, mmd_loss = self.scpoli_model(**batch)

        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        prot_decoding_loss = self.prototype_decoding_loss(x)
        
        # TO DO: recheck this with priginal scpoli
        calc_alpha_coeff = 0.5
        cvae_loss = recon_loss + calc_alpha_coeff * kl_loss + mmd_loss
        
        return x, self.prototypes(x), cvae_loss, prot_decoding_loss

    def get_prototypes(self):
        return self.prototypes.weight.detach().cpu()

    def prototype_distance(self, z: torch.Tensor):
        return torch.cdist(z, self.prototypes.weight)

    def feature_vector_distance(self, z: torch.Tensor):
        return torch.cdist(self.prototypes.weight, z)

    def prototype_decoding_loss(self, z):
        p_dist = self.prototype_distance(z)
        f_dist = self.feature_vector_distance(z)
        return (
            self.reg1 * p_dist.min(1).values.mean()
            + self.reg2 * f_dist.min(1).values.mean()
        )