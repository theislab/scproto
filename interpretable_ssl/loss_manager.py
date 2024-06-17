import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    NT-Xent loss function for contrastive learning.
    
    Parameters:
    - z_i: Tensor of shape (batch_size, projection_dim)
    - z_j: Tensor of shape (batch_size, projection_dim)
    - temperature: A scalar for temperature scaling
    
    Returns:
    - loss: The NT-Xent loss for the batch
    """
    batch_size = z_i.size(0)

    # Normalize the projections
    z = torch.cat([z_i, z_j], dim=0)  # 2N x d
    z = F.normalize(z, dim=1)

    # Similarity matrix (2N x 2N)
    similarity_matrix = torch.matmul(z, z.T)

    # Mask to filter out similarity between augmented pairs
    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z.device)

    # Labels for contrastive loss
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)

    # Remove diagonal elements from similarity matrix and labels
    similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)
    labels = labels[~mask].view(batch_size * 2, -1)

    # Positive pairs (similarity between augmented views of the same image)
    positives = similarity_matrix[labels.bool()].view(batch_size * 2, 1)

    # Negative pairs (all other similarities)
    negatives = similarity_matrix[~labels.bool()].view(batch_size * 2, -1)

    # Concatenate positives and negatives
    logits = torch.cat([positives, negatives], dim=1)

    # Labels for the contrastive loss (positives are the first element in the concatenated tensor)
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(z.device)

    # Compute the cross-entropy loss
    loss = F.cross_entropy(logits / temperature, labels)
    return loss


def repulsion_loss(prototypes):
    """
    Calculate repulsion loss to ensure prototypes are spread out.
    """
    dist_matrix = torch.cdist(prototypes, prototypes, p=2)
    # Invert distances to get repulsion force (1/distance)
    inv_dist_matrix = 1.0 / (dist_matrix + 1e-5)
    repulsion = torch.sum(torch.triu(inv_dist_matrix, diagonal=1))
    return repulsion


class PrototypeLoss:
    def __init__(self) -> None:
        # print('------0.01 vae_reg---------')
        self.vae, self.interpretability = 0, 0
        self.overal = 0
        self.task = 0
        self.vae_reg = 0.01
        self.repulsion_loss, self.repulsion_alpha = 0, 0.005
        self.fixed_values = ['vae_reg', 'fixed_values', 'repulsion_alpha']
        # self.to_norm_loss_keys = ['vae', 'task', 'interpretability']
        # self.max_values = {self.__dict__[key] for key in self.to_norm_loss_keys}
    def calculate_overal(self, vae, interpretability, task=0):
        self.vae = vae
        self.interpretability = interpretability
        self.task = task
        self.overal = self.interpretability + self.vae_reg * self.vae + self.task
        # self.overal = self.vae

    def __add__(self, prot_loss):
        # new_loss = PrototypeLoss()
        for key in self.__dict__:
            if key in self.fixed_values:
                continue
            self.__dict__[key] = prot_loss.__dict__[key] + self.__dict__[key]
        return self

    def normalize(self, data_loader_size):
        for key, val in self.__dict__.items():
            if key in self.fixed_values:
                continue
            self.__dict__[key] = val / data_loader_size
            
        return self

    def set_task_loss(self, task_loss, task_ratio=1):
        self.task = task_loss
        self.overal += self.task * task_ratio
        
    def set_repulsion(self, prototypes):
        self.repulsion_loss = int(repulsion_loss(prototypes))
        self.overal += self.repulsion_alpha * self.repulsion_loss
