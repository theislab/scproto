import torch.nn as nn
import torch
import torch
from sklearn.cluster import KMeans
from scarches.models.scpoli import scPoli

# encoder
# possibly a projection head
# prototype layer


class SwavBase(nn.Module):
    def __init__(
        self, scpoli_encoder, latent_dim, nmb_prototypes #, propagation_reg=0.5, prot_emb_sim_reg=0.5
    ):
        super().__init__()
        self.scpoli_encoder = scpoli_encoder
        self.prototypes = nn.Linear(latent_dim, nmb_prototypes, bias=False)
        self.projection_head = None
        self.l2norm = True
        # self.propagation_reg = propagation_reg
        # self.prot_emb_sim_reg = prot_emb_sim_reg

    def init_prototypes_kmeans(self, embeddings, nmb_prots):
        # Run KMeans on embeddings (convert to numpy for compatibility)
        kmeans = KMeans(n_clusters=nmb_prots)
        kmeans.fit(embeddings.cpu().numpy())

        # Get cluster centers and convert them back to a PyTorch tensor
        cluster_centers = torch.tensor(kmeans.cluster_centers_)
        self.set_prototypes(cluster_centers)

    def forward(self, batch):
        x, recon_loss, kl_loss, mmd_loss = self.scpoli_encoder(**batch)

        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        prot_decoding_loss = self.prototype_decoding_loss(x)

        # TO DO: recheck this with original scpoli
        calc_alpha_coeff = 0.5
        cvae_loss = recon_loss + calc_alpha_coeff * kl_loss + mmd_loss

        # return 2 x so it would be match the other model output
        return x, x, self.prototypes(x), cvae_loss, prot_decoding_loss

    def propagation(self, z: torch.Tensor):
        dist = torch.cdist(z, self.prototypes.weight)
        return dist.min(1).values.mean()
    def embedding_similarity(self, z: torch.Tensor):
        dist = torch.cdist(self.prototypes.weight, z)
        return dist.min(1).values.mean()
    def prototype_decoding_loss(self, z):
        return self.propagation(z), self.embedding_similarity(z)

    def set_scpoli_encoder(self, scpoli_encoder):
        self.scpoli_encoder = scpoli_encoder

    def encode(self, batch):
        encoder_out, x, x_mapped, _, _ = self.forward(batch)
        return encoder_out, x, x_mapped

    def get_prototypes(self):
        return self.prototypes.weight.data

    def normalize_prototypes(self):
        w = self.get_prototypes().clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        self.set_prototypes(w)

    def set_prototypes(self, w):
        with torch.no_grad():
            self.prototypes.weight.copy_(w)

class SwAVModel(SwavBase):
    def __init__(self, latent_dim, nmb_prototypes, adata): # , propagation_reg=0.5, prot_emb_sim_reg=0.5
        # self.cell_type_key = "cell_type"
        self.condition_key = "study"
        self.scpoli_ = self.init_scpoli(adata, latent_dim)
        super().__init__(self.scpoli_.model, latent_dim, nmb_prototypes) # , propagation_reg, prot_emb_sim_reg
        
    def init_scpoli(self, adata, latent_dim):
        return scPoli(
            adata=adata,
            condition_keys=self.condition_key,
            # cell_type_keys=self.cell_type_key,
            latent_dim=latent_dim,
            recon_loss="nb",
        )
# def get_projector(input_dim, hidden_dim, output_dim):
#     return nn.Sequential(
#         nn.Linear(input_dim, hidden_dim),
#         nn.BatchNorm1d(hidden_dim),
#         nn.ReLU(inplace=True),
#         nn.Linear(hidden_dim, output_dim),
#     )


# class SwavModel(SwavBase):
#     def __init__(
#         self,
#         scpoli_model,
#         latent_dim,
#         nmb_prototypes,
#         use_projector=False,
#         hidden_mlp=None,
#         swav_dim=None,
#     ):
#         super().__init__(scpoli_model, latent_dim, nmb_prototypes)

#         if use_projector:
#             self.projection_head = get_projector(latent_dim, hidden_mlp, latent_dim)
#             self.prototype_projector = get_projector(swav_dim, hidden_mlp, latent_dim)
#             self.prototypes = nn.Parameter(
#                 self.get_nn_linear_weights(nmb_prototypes, swav_dim), requires_grad=True
#             )
#         else:
#             self.projection_head = None
#             self.prototype_projector = None
#             self.prototypes = nn.Parameter(
#                 self.get_nn_linear_weights(nmb_prototypes, latent_dim), requires_grad=True
#             )

#     def get_nn_linear_weights(self, input_size, output_size):
#         # Create an nn.Linear layer without a bias term
#         linear_layer = nn.Linear(input_size, output_size, bias=False)

#         # Retrieve the initialized weights from the nn.Linear layer
#         weight = linear_layer.weight.detach().clone()
#         return weight.t()

#     def forward(self, batch):
#         encoder_out, recon_loss, kl_loss, mmd_loss = self.scpoli_model(**batch)

#         # TO DO: recheck this with priginal scpoli
#         calc_alpha_coeff = 0.5
#         cvae_loss = recon_loss + calc_alpha_coeff * kl_loss + mmd_loss
#         prot_decoding_loss = self.prototype_decoding_loss(encoder_out)

#         if self.projection_head is not None:
#             x = self.projection_head(encoder_out)
#         else:
#             x = encoder_out

#         if self.l2norm:
#             x = nn.functional.normalize(x, dim=1, p=2)

#         x_mapped = torch.matmul(x, self.prototypes.t())

#         # return x, self.prototypes(x), cvae_loss, prot_decoding_loss
#         return encoder_out, x, x_mapped, cvae_loss, prot_decoding_loss

#     def get_prototypes(self):
#         return self.prototypes.data

#     def get_interpretable_prototypes(self):
#         if self.prototype_projector is not None:
#             return self.prototype_projector(self.prototypes)
#         return self.prototypes

#     def set_prototypes(self, prototypes):
#         return self.prototypes.copy_(prototypes)

#     def prototype_distance(self, z: torch.Tensor):
#         return torch.cdist(z, self.get_interpretable_prototypes())

#     def feature_vector_distance(self, z: torch.Tensor):
#         return torch.cdist(self.get_interpretable_prototypes(), z)

#     def prototype_decoding_loss(self, z):
#         p_dist = self.prototype_distance(z)
#         f_dist = self.feature_vector_distance(z)
#         return (
#             self.reg1 * p_dist.min(1).values.mean()
#             + self.reg2 * f_dist.min(1).values.mean()
#         )
