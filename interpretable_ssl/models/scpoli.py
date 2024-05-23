from interpretable_ssl.models.autoencoder import PrototypeVAE
from interpretable_ssl.models.prototype_model import *

from interpretable_ssl.models.prototype_barlow import BarlowProjector
from scarches.models.scpoli import scPoli
import scarches.trainers.scpoli._utils as scpoli_utils

from copy import deepcopy
import random
from interpretable_ssl import train_utils

def get_scpoli(adata, latent_dim):
    condition_key = "study"
    cell_type_key = "cell_type"
    return scPoli(
        adata=adata,
        condition_keys=[condition_key],
        cell_type_keys=[cell_type_key],
        latent_dim=latent_dim,
        recon_loss="nb",
    )
def to_device(scpoli_batch, device):
    return {key: scpoli_batch[key].to(device) for key in scpoli_batch}

def scpoli_loss(model, scpoli_batch, calc_alpha_coeff=0.5):
        z, recon_loss, kl_loss, mmd_loss = model(**scpoli_batch)
        cvae_loss = recon_loss + calc_alpha_coeff * kl_loss + mmd_loss
        return cvae_loss, z
    

class PrototypeScpoli(nn.Module):
    def __init__(
        self, adata, latent_dim, num_prototypes, head
    ) -> None:
        super().__init__()
        self.scpoli = get_scpoli(adata, latent_dim)
        self.scpoli_model = self.scpoli.model
        self.prototype_head = PrototypeBase(num_prototypes, latent_dim)
        self.head = head

        # TO DO
        self.calc_alpha_coeff = 0.5
        self.device = utils.get_device()

    def to_device(self, scpoli_batch):
        return to_device(scpoli_batch, self.device)

    def prototype_forward(self, scpoli_batch):
        
        cvae_loss, z = scpoli_loss(self.scpoli_model, scpoli_batch)
        z = z.to(self.device)
        interpretablity_loss = self.prototype_head.forward(z)
        prot_loss = PrototypeLoss()
        prot_loss.calculate_overal(cvae_loss, interpretablity_loss)
        prot_dist = self.prototype_head.prototype_distance(z)
        return prot_loss, prot_dist

    def ssl_forward(self, scpoli_batch1, scpoli_batch2):
        scpoli_batch1, scpoli_batch2 = self.to_device(scpoli_batch1), self.to_device(scpoli_batch2)
        prot_loss1, prot_dist1 = self.prototype_forward(scpoli_batch1)
        prot_loss2, prot_dist2 = self.prototype_forward(scpoli_batch2)

        head_loss = self.head(prot_dist1, prot_dist2)

        batch_loss = prot_loss1 + prot_loss2
        batch_loss.set_task_loss(head_loss)
        return batch_loss
    def get_representation(self, adata):
        return self.scpoli.get_latent(adata, mean=True)
    
    def get_prototypes(self):
        return self.prototype_head.prototype_vectors
    
    def decode_prototypes(self, batch_embeddings):
        prot_cells, _ = self.scpoli_model.decoder(self.get_prototypes(), batch_embeddings)
        return prot_cells.detach().numpy()
    
    def find_prototypes_closest_cell_idx(self, cell_embeddings):
        dist = torch.cdist(self.get_prototypes(), torch.tensor(cell_embeddings))
        return dist.argmin(dim=1)
        
    def calc_batch_embedding(self, inp):
        batch = inp['batch']
        batch_embeddings = torch.hstack([self.scpoli_model.embeddings[i](batch[:, i]) for i in range(batch.shape[1])])
        return batch_embeddings

    def get_closest_cell_batch_embedding(self, train_adata):
        cell_embeddings = self.get_representation(train_adata)
        closest_cell_idx = self.find_prototypes_closest_cell_idx(cell_embeddings)
        closest_cells = train_adata[closest_cell_idx.numpy()]
        loader = generate_scpoli_dataloder(closest_cells, self.scpoli_model)
        batch_embeddings = self.calc_batch_embedding(next(iter(loader)))
        return batch_embeddings
        

    def get_all_batch_embeddings(self):
        all_batches = range(len(self.scpoli_model.condition_encoders['study']))
        inp = {'batch':torch.tensor([i for i in all_batches]).reshape(-1, 1)}
        return self.calc_batch_embedding(inp)
    
    
    def decode_prototypes_using_closest_cell(self, train_adata):
        closest_cell_batch_embedding = self.get_closest_cell_batch_embedding(train_adata)
        return self.decode_prototypes(closest_cell_batch_embedding)
    
    def decode_prototypes_using_all_batch(self):
        all_batch_embeddings = self.get_all_batch_embeddings()
        
        cells = [self.decode_prototypes(batch_emb.reshape(1, -1).repeat(16, 1)) for batch_emb in all_batch_embeddings]
        cells = torch.tensor(cells)
        cells = cells.mean(dim=0)
        return cells
    
class LinearPrototypeScpoli(PrototypeScpoli):
    def __init__(self, adata, latent_dim, num_prototypes, head) -> None:
        super().__init__(adata, latent_dim, num_prototypes, head)
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, scpoli_batch) -> PrototypeLoss:
        scpoli_batch = self.to_device(scpoli_batch)
        prot_loss, prot_dist = self.prototype_forward(scpoli_batch)
        batch_loss = prot_loss
        
        head_out = self.head(prot_dist)
        y = scpoli_batch['celltypes']
        head_loss = self.loss(head_out, y.squeeze())
        batch_loss.set_task_loss(head_loss, 10)
        return batch_loss
    


class BarlowPrototypeScpoli(PrototypeScpoli):
    def __init__(
        self, adata, latent_dim, num_prototypes
    ) -> None:
        
        projection_sizes, lambd = [num_prototypes, 32, 32, 32], 3.9e-3
        barlow_model = BarlowProjector(projection_sizes, lambd)
        super().__init__(adata, latent_dim, num_prototypes, barlow_model)


    def forward(self, scpoli_batch1, scpoli_batch2):
        return self.ssl_forward(scpoli_batch1, scpoli_batch2)
    
class BarlowScpoli(nn.Module):
    def __init__(self, adata, latent_dim) -> None:
        super().__init__()
        self.scpoli = get_scpoli(adata, latent_dim)
        self.cvae = self.scpoli.model
        projection_sizes, lambd = [latent_dim, 32, 32, 32], 3.9e-3
        self.barlow = BarlowProjector(projection_sizes, lambd)
        self.device = utils.get_device()
    
    def forward(self, scpoli_batch1, scpoli_batch2):
        scpoli_batch1, scpoli_batch2 = to_device(scpoli_batch1, self.device), to_device(scpoli_batch2, self.device)
        cvae_loss1, z1 = scpoli_loss(self.cvae, scpoli_batch1)
        cvae_loss2, z2 = scpoli_loss(self.cvae, scpoli_batch2)
        
        loss = PrototypeLoss()
        loss.calculate_overal(cvae_loss1 + cvae_loss2, 0)
        
        barlow_loss = self.barlow(z1, z2)
        loss.set_task_loss(barlow_loss)
        return loss
    def get_representation(self, adata):
        return self.scpoli.get_latent(adata, mean=True)
    
class AdataAugmenter:
    def __init__(self, adata) -> None:
        self.adata = adata
        self.cell_type_matched = {}

    def get_macthed_idx(self, cell_type):
        if cell_type in self.cell_type_matched:
            return self.cell_type_matched[cell_type]
        matched_rows = self.adata.obs.cell_type == cell_type
        matched_idx = [idx for idx, macthed in enumerate(matched_rows) if macthed]
        self.cell_type_matched[cell_type] = matched_idx
        return matched_idx

    def augment_cell_type(self, cell_type):
        matched_idx = self.get_macthed_idx(cell_type)
        return random.sample(matched_idx, 1)[0]

    def augment(self):
        augmented = deepcopy(self.adata)
        cell_types = self.adata.obs.cell_type
        augmented_idx = [self.augment_cell_type(cell_type) for cell_type in cell_types]
        return augmented[augmented_idx]

def generate_scpoli_dataset(adata, scpoli_model):
    condition_key = "study"
    cell_type_key = "cell_type"
    train, _ = scpoli_utils.make_dataset(
        adata,
        1,
        condition_keys=[condition_key],
        cell_type_keys=[cell_type_key],
        condition_encoders=scpoli_model.condition_encoders,
        conditions_combined_encoder=scpoli_model.conditions_combined_encoder,
        cell_type_encoder=scpoli_model.cell_type_encoder,
    )
    return train
def generate_scpoli_dataloder(adata, scpoli_model, batch_size = 16):
    train = generate_scpoli_dataset(adata, scpoli_model)
    loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, collate_fn=scpoli_utils.custom_collate, shuffle=True
    )
    return loader


def prepare_augmented_data(adata, model):
    augmented_adata = AdataAugmenter(adata).augment()
    scpoli_model = model.scpoli.model
    augmented_loader = generate_scpoli_dataloder(augmented_adata, scpoli_model)
    return augmented_loader


def train_step(model: PrototypeScpoli, train_adata, train_loader, optimizer):
    augmented_loader = prepare_augmented_data(train_adata, model)
    return train_utils.train_step(
        model, zip(train_loader, augmented_loader), optimizer, len(train_adata)
    )


def test_step(model, test_loader, test_adata):
    augmented_loader = prepare_augmented_data(test_adata, model)
    return train_utils.test_step(
        model, zip(test_loader, augmented_loader), len(test_adata)
    )
