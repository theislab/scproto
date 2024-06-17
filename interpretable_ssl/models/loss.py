import torch
import torch.nn.functional as F

def scpoli_loss(model, scpoli_batch, calc_alpha_coeff=0.5):
        z, recon_loss, kl_loss, mmd_loss = model(**scpoli_batch)
        cvae_loss = recon_loss + calc_alpha_coeff * kl_loss + mmd_loss
        return cvae_loss, z
    
# simclr loss
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T)
    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(z.device)
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
    similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)
    labels = labels[~mask].view(batch_size * 2, -1)
    positives = similarity_matrix[labels.bool()].view(batch_size * 2, 1)
    negatives = similarity_matrix[~labels.bool()].view(batch_size * 2, -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(z.device)
    return F.cross_entropy(logits / temperature, labels)