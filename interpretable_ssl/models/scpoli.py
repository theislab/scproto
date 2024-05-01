from interpretable_ssl.models.autoencoder import PrototypeVAE
from interpretable_ssl.models.prototype_model import *

from interpretable_ssl.models.prototype_barlow import BarlowProjector
from scarches.models.scpoli import scPoli
import scarches.trainers.scpoli._utils as scpoli_utils

from copy import deepcopy
import random
from interpretable_ssl import train_utils


class PrototypeScpoli(nn.Module):
    def __init__(
        self, adata, latent_dim, num_prototypes, train_only_scpoli=False
    ) -> None:
        super().__init__()
        condition_key = "study"
        cell_type_key = "cell_type"
        self.scpoli = scPoli(
            adata=adata,
            condition_keys=[condition_key],
            cell_type_keys=[cell_type_key],
            latent_dim=latent_dim,
            recon_loss="nb",
        )
        self.scpoli_model = self.scpoli.model
        self.prototype_head = PrototypeBase(num_prototypes, latent_dim)
        projection_sizes, lambd = [num_prototypes, 32, 32, 32], 3.9e-3
        self.barlow_model = BarlowProjector(projection_sizes, lambd)

        # TO DO
        self.calc_alpha_coeff = 0.5
        self.device = utils.get_device()

    def to_device(self, scpoli_batch):
        return {key: scpoli_batch[key].to(self.device) for key in scpoli_batch}

    def prototype_forward(self, scpoli_batch):
        z, recon_loss, kl_loss, mmd_loss = self.scpoli_model(**scpoli_batch)
        cvae_loss = recon_loss + self.calc_alpha_coeff * kl_loss + mmd_loss
        z = z.to(self.device)
        interpretablity_loss = self.prototype_head.forward(z)
        prot_loss = PrototypeLoss()
        prot_loss.calculate_overal(cvae_loss, interpretablity_loss)
        prot_dist = self.prototype_head.prototype_distance(z)
        return prot_loss, prot_dist

    def forward(self, scpoli_batch1, scpoli_batch2):
        scpoli_batch1, scpoli_batch2 = self.to_device(scpoli_batch1), self.to_device(scpoli_batch2)
        prot_loss1, prot_dist1 = self.prototype_forward(scpoli_batch1)
        prot_loss2, prot_dist2 = self.prototype_forward(scpoli_batch2)

        barlow_loss = self.barlow_model(prot_dist1, prot_dist2)

        batch_loss = prot_loss1 + prot_loss2
        batch_loss.set_task_loss(barlow_loss)
        return batch_loss


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


def generate_scpoli_dataloder(adata, scpoli_model):
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
    loader = torch.utils.data.DataLoader(
        train, batch_size=16, collate_fn=scpoli_utils.custom_collate, shuffle=True
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
