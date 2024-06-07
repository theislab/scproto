from interpretable_ssl.trainers.trainer import *
import scvi


class ScviTrainer(Trainer):
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
        return f"scvi-latent_dim{self.latent_dims}"

    def get_model_path(self):
        path = super().get_model_path()
        name, _ = os.path.splitext(path)
        return name

    def load_query_model(self):
        return scvi.model.SCVI.load_query_data(
            self.query.adata,
            self.get_model_path(),
        )

    def get_ref_query_latent(self):
        # Setup AnnData for scVI using the same settings as the reference data
        model = self.load_query_model()
        
        # Fine-tune the model on the query dataset
        model.train(max_epochs=self.fine_tuning_epochs, use_gpu=(self.device == 'cuda'))
        query_latent = model.get_latent_representation(self.query.adata)
        ref_latent = model.get_latent_representation(self.ref.adata)
        all_latent = model.get_latent_representation(self.dataset.adata)
        return ref_latent, query_latent, all_latent
