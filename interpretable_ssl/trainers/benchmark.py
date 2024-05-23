from scarches.models.scpoli import scPoli
from interpretable_ssl import utils

# get adata
# initialize scPoli with same parameters
# train
# save


def train_scpoli(adata, latent_dim, dataset_name):
    condition_key = "study"
    cell_type_key = "cell_type"
    scpoli_trainer = scPoli(
        adata=adata,
        condition_keys=condition_key,
        cell_type_keys=cell_type_key,
        latent_dim=latent_dim,
        recon_loss="nb",
    )

    scpoli_trainer.train(
        n_epochs=100,
        pretraining_epochs=50,
    )
    model_path = f"{utils.get_model_dir} / {dataset_name} / original-scpoli-latent{latent_dim}.pth"
    print(model_path)
    utils.save_model(scpoli_trainer.model, model_path)
