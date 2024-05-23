from interpretable_ssl.models.autoencoder import PrototypeVAE

import scvi
from interpretable_ssl.models.scvi_constants import MODULE_KEYS
import torch
from scvi.train import TrainingPlan

class ScviVAE(PrototypeVAE):
    def __init__(self, latent_dims, adata) -> None:
        super().__init__()
        self.latent_dims = latent_dims
        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="batch")
        self.scvi_model = scvi.model.SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=latent_dims)
        self.training_plan = TrainingPlan(self.scvi_model.module)

    def get_latent_dims(self):
        return self.latent_dims

    def encode(self, x,  mc_samples: int = 5000):
        inference_inputs = self.scvi_model.module._get_inference_input(x)
        outputs = self.scvi_model.module.inference(**inference_inputs)
        if MODULE_KEYS.QZ_KEY in outputs:
            qz = outputs[MODULE_KEYS.QZ_KEY]
        else:
            qz_m, qz_v = outputs[MODULE_KEYS.QZM_KEY], outputs[MODULE_KEYS.QZV_KEY]
            qz = torch.distributions.Normal(qz_m, qz_v.sqrt())
            # does each model need to have this latent distribution param?
        if self.scvi_model.module.latent_distribution == "ln":
            samples = qz.sample([mc_samples])
            z = torch.nn.functional.softmax(samples, dim=-1)
            z = z.mean(dim=0)
        else:
            z = qz.loc
        return z

    def decode(self, z):
        generative_inputs = self.scvi_model.module._get_generative_input(z)
        out = self.scvi_model.module.generative(**generative_inputs)
        return out

    def calculate_loss(self, x):
        return self.training_plan.training_step(x)
    