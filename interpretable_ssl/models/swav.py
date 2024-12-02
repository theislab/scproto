import torch.nn as nn
import torch
import torch
from sklearn.cluster import KMeans
from scarches.models.scpoli import scPoli
from scarches.models.scpoli._utils import one_hot_encoder
from scarches.models.trvae.losses import nb

import torch.nn.functional as F

# encoder
# possibly a projection head
# prototype layer


class SwavBase(nn.Module):
    def __init__(
        self,
        scpoli_encoder,
        latent_dim,
        nmb_prototypes,  # , propagation_reg=0.5, prot_emb_sim_reg=0.5
        multi_layer_proto=False,
        np2=None,
    ):
        super().__init__()
        self.scpoli_encoder = scpoli_encoder
        self.prototypes = nn.Linear(latent_dim, nmb_prototypes, bias=False)
        if multi_layer_proto:
            print("initializing cell proto layer")
            self.cell_protos = nn.Linear(latent_dim, np2, bias=False)
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

    def compute_cvae_loss(self, recon_loss, kl_loss, mmd_loss):
        calc_alpha_coeff = 0.5
        cvae_loss = recon_loss + calc_alpha_coeff * kl_loss + mmd_loss
        return cvae_loss

    def encoder_out(self, batch):
        x, recon_loss, kl_loss, mmd_loss = self.scpoli_encoder(**batch)
        cvae_loss = self.compute_cvae_loss(recon_loss, kl_loss, mmd_loss)
        return x, cvae_loss

    def forward(self, batch):
        # x, recon_loss, kl_loss, mmd_loss = self.scpoli_encoder(**batch)
        x, cvae_loss = self.encoder_out(batch)

        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        prot_decoding_loss = self.prototype_decoding_loss(x)

        # TO DO: recheck this with original scpoli
        # calc_alpha_coeff = 0.5
        # cvae_loss = recon_loss + calc_alpha_coeff * kl_loss + mmd_loss

        # return 2 x so it would be match the other model output
        if hasattr(self, "cell_protos"):
            return (
                x,
                x,
                (self.prototypes(x), self.cell_protos(x)),
                cvae_loss,
                prot_decoding_loss,
            )
        else:
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
        if hasattr(self, "cell_protos"):
            wc = self.cell_protos.weight.data
            wc = nn.functional.normalize(wc, dim=1, p=2)
            with torch.no_grad():
                self.cell_protos.weight.copy_(wc)

    def set_prototypes(self, w):
        with torch.no_grad():
            self.prototypes.weight.copy_(w)

    def prototypes_avg_distance(self):
        """
        Calculate the average of the average distances for each tensor in a (p, d) tensor.

        Args:
            tensor (torch.Tensor): Input tensor of shape (p, d).

        Returns:
            float: The average of the average distances for all p tensors.
        """
        tensor = self.get_prototypes()
        # Calculate pairwise distances using broadcasting
        pairwise_diff = tensor.unsqueeze(1) - tensor.unsqueeze(0)  # Shape: (p, p, d)
        pairwise_distances = torch.norm(pairwise_diff, dim=2)  # Shape: (p, p)

        # Average distance for each tensor (exclude self-distance by setting diagonal to 0)
        pairwise_distances.fill_diagonal_(0)
        avg_distances_per_tensor = pairwise_distances.sum(dim=1) / (tensor.shape[0] - 1)

        # Average of these distances
        overall_avg_distance = avg_distances_per_tensor.mean().item()

        return overall_avg_distance


class SwAVModel(SwavBase):
    def __init__(
        self, latent_dim, nmb_prototypes, adata, multi_layer_proto=False, np2=None
    ):  # , propagation_reg=0.5, prot_emb_sim_reg=0.5
        # self.cell_type_key = "cell_type"
        self.condition_key = "study"
        self.scpoli_ = self.init_scpoli(adata, latent_dim)
        super().__init__(
            self.scpoli_.model, latent_dim, nmb_prototypes, multi_layer_proto, np2
        )  # , propagation_reg, prot_emb_sim_reg

    def init_scpoli(self, adata, latent_dim):
        return scPoli(
            adata=adata,
            condition_keys=self.condition_key,
            # cell_type_keys=self.cell_type_key,
            latent_dim=latent_dim,
            recon_loss="nb",
        )


class SwAVDecodableProto(SwAVModel):

    def find_closest_prototype(self, embeddings):
        similarity_scores = self.prototypes(embeddings)
        # Get the index of the most similar prototype (highest similarity)
        closest_prototype_indices = torch.argmax(
            similarity_scores, dim=1
        )  # Shape: (b,)

        # Retrieve the corresponding prototype vectors
        prototype_vectors = self.prototypes.weight  # Shape: (num_prototypes, input_dim)
        closest_prototypes = prototype_vectors[
            closest_prototype_indices
        ]  # Shape: (b, input_dim)

        return closest_prototypes

    def compute_recon_loss(self, x, outputs, sizefactor, combined_batch):
        # nb
        dec_mean_gamma, y1 = outputs
        size_factor_view = sizefactor.unsqueeze(1).expand(
            dec_mean_gamma.size(0), dec_mean_gamma.size(1)
        )
        dec_mean = dec_mean_gamma * size_factor_view
        dispersion = F.linear(
            one_hot_encoder(combined_batch, self.scpoli_encoder.n_conditions_combined),
            self.scpoli_encoder.theta,
        )
        dispersion = torch.exp(dispersion)

        recon_loss = -nb(x=x, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()
        return recon_loss

    def decode_prototypes(self, prototypes, data_dict):
        batch_embeddings = torch.hstack(
            [
                self.scpoli_encoder.embeddings[i](data_dict["batch"][:, i])
                for i in range(data_dict["batch"].shape[1])
            ]
        )
        outputs = self.scpoli_encoder.decoder(prototypes, batch_embeddings)
        loss = self.compute_recon_loss(
            data_dict["x"],
            outputs,
            data_dict["sizefactor"],
            data_dict["combined_batch"],
        )
        return loss

    def calculate_proto_recon_loss(self, embeddings, data_dict):
        closest_prototypes = self.find_closest_prototype(embeddings)
        loss = self.decode_prototypes(closest_prototypes, data_dict)
        return loss

    def encoder_out(self, batch):
        x, recon_loss, kl_loss, mmd_loss = self.scpoli_encoder(**batch)
        proto_recon_loss = self.calculate_proto_recon_loss(x, batch)
        cvae_loss = self.compute_cvae_loss(proto_recon_loss, kl_loss, mmd_loss)
        return x, cvae_loss
