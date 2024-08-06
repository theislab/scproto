from interpretable_ssl.trainers.trainer import *
import scvi


class ScviTrainer(Trainer):
    def __init__(
        self, partially_train_ratio=None, self_supervised=False, dataset=None
    ) -> None:
        super().__init__(partially_train_ratio, self_supervised, dataset)
        # add batch_size in name in version 2
        self.model_name_version = 2

    def get_model(self):
        scvi.model.SCVI.setup_anndata(self.ref.adata, layer="counts", batch_key="batch")
        vae = scvi.model.SCVI(
            self.ref.adata, gene_likelihood="nb", n_layers=2, n_latent=self.latent_dims
        )
        return vae

    def save_model(self, model):
        model.save(self.get_model_path(), overwrite=True)

    def load_model(self):
        return scvi.model.SCVI.load(self.get_model_path(), self.ref.adata)

    def train(self, epochs):
        model = self.get_model()
        model.train(epochs)
        self.save_model(model)

    def get_model_name(self):
        name = f"scvi-latent_dim{self.latent_dims}"
        name = self.append_batch(name)
        return name

    def get_model_path(self):
        path = super().get_model_path()
        name, _ = os.path.splitext(path)
        return name

    def get_query_model(self):
        return scvi.model.SCVI.load_query_data(
            self.query.adata,
            self.get_model_path(),
        )
        
    def load_query_model(self, model, path):
        return scvi.model.SCVI.load(path, self.query.adata)
    
    def finetune_query_model(self, model):
        model.train(max_epochs=self.fine_tuning_epochs, use_gpu=(self.device == "cuda"))
        model.save(self.get_query_model_path(), overwrite=True)
        return model

    def get_query_model_latent(self, model, adata):
        return model.get_latent_representation(adata)
