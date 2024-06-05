import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class PrototypeBase(nn.Module):
    def __init__(self, num_prototypes, latent_dims, latent_space) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_shape = (self.num_prototypes, latent_dims)
        self.prototype_vectors = nn.Parameter(
            self.get_initialize_prototypes(latent_space), requires_grad=True
        )
        self.reg1 = 0.05
        self.reg2 = 0.05
        self.vae_reg = 0.5

    def prototype_distance(self, z: torch.Tensor):
        return torch.cdist(z, self.prototype_vectors)

    def feature_vector_distance(self, z: torch.Tensor):
        return torch.cdist(self.prototype_vectors, z)

    def get_initialize_prototypes(self, latent_space):
        kmeans = KMeans(n_clusters=self.num_prototypes, random_state=0).fit(latent_space)
        initial_prototypes = kmeans.cluster_centers_
        
        return torch.tensor(initial_prototypes, dtype=torch.float32)

    def forward(self, z):
        p_dist = self.prototype_distance(z)
        f_dist = self.feature_vector_distance(z)
        return (
            self.reg1 * p_dist.min(1).values.mean()
            + self.reg2 * f_dist.min(1).values.mean()
        )
