import random
from copy import deepcopy
import scarches.trainers.scpoli._utils as scpoli_utils
import torch

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
def generate_scpoli_dataloder(adata, scpoli_model, sampler = None, batch_size = 16):
    train = generate_scpoli_dataset(adata, scpoli_model)
    loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, collate_fn=scpoli_utils.custom_collate, sampler=sampler
    )
    return loader

def prepare_scpoli_dataloader(adata, model, batch_size, augment= False):
    if augment:
        adata = AdataAugmenter(adata).augment()
    scpoli_model = model.scpoli.model
    return generate_scpoli_dataloder(adata, scpoli_model, batch_size=batch_size)
