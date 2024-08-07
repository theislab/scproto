import torch.nn as nn
import torch

# encoder
# possibly a projection head
# prototype layer

class SwavBase(nn.Module):
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
        
        # return 2 x so it would be match the other model output
        return x, x, self.prototypes(x), cvae_loss, prot_decoding_loss

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
    
    def set_scpoli_model(self, scpoli_model):
        self.scpoli_model = scpoli_model
        
    def encode(self, batch):
        encoder_out, x, x_mapped, _, _ = self.forward(batch)
        return encoder_out, x, x_mapped
    
    def get_prototypes(self):
        return self.prototypes.weight.data
    
    def normalize_prototypes(self):
        prototypes = self.get_prototypes().clone()
        prototypes = nn.functional.normalize(prototypes, dim=1, p=2)
        self.set_prototypes(prototypes)
        
    def set_prototypes(self, prototypes):
        self.prototypes.weight.copy_(prototypes)
        
class SwavModel(SwavBase):
    def __init__(
        self,
        scpoli_model,
        latent_dim,
        nmb_prototypes,
        use_projector=False,
        hidden_mlp=None,
        swav_dim=None,
    ):
        super().__init__(scpoli_model, latent_dim, nmb_prototypes)
        
        self.prototypes = nn.Parameter(
            torch.randn(nmb_prototypes, latent_dim), requires_grad=True
        )

        if use_projector:
            self.projection_head = nn.Sequential(
                nn.Linear(latent_dim, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, swav_dim),
            )
        else:
            self.projection_head = None
            
    def forward(self, batch):
        encoder_out, recon_loss, kl_loss, mmd_loss = self.scpoli_model(**batch)

        # TO DO: recheck this with priginal scpoli
        calc_alpha_coeff = 0.5
        cvae_loss = recon_loss + calc_alpha_coeff * kl_loss + mmd_loss

        # interpretablity loss
        prot_decoding_loss = self.prototype_decoding_loss(encoder_out)

        if self.projection_head is not None:
            x = self.projection_head(encoder_out)
            prototypes = self.projection_head(self.prototypes)
        else:
            x = encoder_out
            prototypes = self.prototypes

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)
            # prototypes = nn.functional.normalize(prototypes, dim=1, p=2)

        x_mapped = torch.matmul(x, prototypes.t())

        # return x, self.prototypes(x), cvae_loss, prot_decoding_loss
        return encoder_out, x, x_mapped, cvae_loss, prot_decoding_loss

    def get_prototypes(self):
        return self.prototypes.data
    
    def set_prototypes(self, prototypes):
        return self.prototypes.copy_(prototypes)

    def prototype_distance(self, z: torch.Tensor):
        return torch.cdist(z, self.get_prototypes())

    def feature_vector_distance(self, z: torch.Tensor):
        return torch.cdist(self.get_prototypes(), z)

    def prototype_decoding_loss(self, z):
        p_dist = self.prototype_distance(z)
        f_dist = self.feature_vector_distance(z)
        return (
            self.reg1 * p_dist.min(1).values.mean()
            + self.reg2 * f_dist.min(1).values.mean()
        )
