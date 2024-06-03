from interpretable_ssl.trainers.trainer import Trainer
from interpretable_ssl.models.scpoli import *
from sklearn.model_selection import train_test_split
from interpretable_ssl.immune.dataset import ImmuneDataset
from interpretable_ssl.train_utils import optimize_model
import sys
from interpretable_ssl.evaluation.visualization import plot_umap
from torch.profiler import profile, record_function, ProfilerActivity


class ScpoliTrainer(Trainer):
    def __init__(self, dataset=None) -> None:
        super().__init__(dataset=dataset)
        
        self.latent_dims = 8
        self.num_prototypes = 16
        self.experiment_name = "scpoli"
        self.train_adata = None
        self.val_adata = None
        self.best_val_loss = sys.maxsize
        self.ref, self.query = self.dataset.get_train_test()

    def split_train_test(self, ref):
        train_idx, val_idx = train_test_split(range(len(ref.adata)))
        train, val = ref.adata[train_idx], ref.adata[val_idx]
        return train, val
    def get_dataset(self):
        return ImmuneDataset()

    def get_model(self, adata):
        return BarlowPrototypeScpoli(adata, self.latent_dims, self.num_prototypes)

    def prepare_scpoli_data_splits(self, ref, scpoli_model):
        train_adata, val_adata = self.split_train_test(ref)
        train_loader = generate_scpoli_dataloder(train_adata, scpoli_model)
        val_loader = generate_scpoli_dataloder(val_adata, scpoli_model)
        return train_adata, train_loader, val_adata, val_loader

    def to_save(self, val_loss):
        if val_loss.overal < self.best_val_loss:
            self.best_val_loss = val_loss.overal
            return True
        return False

    def train_step(self, model, train_loader, optimizer):
        return train_step(model, self.train_adata, train_loader, optimizer)

    def test_step(self, model, val_loader):
        return test_step(model, val_loader, self.val_adata)
    
    def train(self, epochs):
        # prepare data (train, test)
        # define model
        # init training params + wandb
        # train loop:
        #   for each epoch:
        #      calculate loss
        #      optimize
        #      log loss

        ref, query = self.ref, self.query
        model = self.get_model(ref.adata)
        model.to(self.device)

        train_adata, train_loader, val_adata, val_loader = (
            self.prepare_scpoli_data_splits(ref, model.scpoli.model)
        )

        self.train_adata, self.val_adata = train_adata, val_adata

        # init training parameter and wandb
        optimizer = self.get_optimizer(model)
        model_path = self.get_model_path()

        # init wandb
        self.init_wandb(model_path, len(ref), len(query))
        self.best_val_loss = sys.maxsize

        for epoch in range(epochs):
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:

                train_loss = self.train_step(model, train_loader, optimizer)
                val_loss = self.test_step(model, val_loader)
                self.log_loss(train_loss, val_loss)
                if self.to_save(val_loss):
                    utils.save_model_checkpoint(model, optimizer, epoch, model_path)
                prof.export_chrome_trace("trace.json")
        return train_loss.overal, self.best_val_loss

    def load_model(self):

        model = self.get_model(self.ref.adata)
        path = self.get_model_path()
        model.load_state_dict(torch.load(path)["model_state_dict"])
        return model

    def get_representation(self, model, adata):
        return model.get_representation(adata)

    def visualize(self):
        model = self.load_model()
        model.to(self.device)
        # get latent representation of reference data
        data_latent_source = self.get_representation(model, self.ref.adata)
        self.ref.adata.obsm[f"{self.experiment_name}"] = data_latent_source
        
        plot_umap(self.ref.adata, f"{self.experiment_name}")

    def get_query_model(self):
        model = self.load_model()
        scpoli_query = scPoli.load_query_data(
            adata=self.query.adata,
            reference_model=model.scpoli,
            labeled_indices=[],
        )
        scpoli_query.train(
            n_epochs=50,
            pretraining_epochs=40,
            eta=10
        )
        return scpoli_query
    
    def get_query_latent(self):
        scpoli_query = self.get_query_model()
        query_latent = scpoli_query.get_latent(
            self.query.adata, 
            mean=True
        )
        return query_latent
        
class SslTrainer(ScpoliTrainer):
    def __init__(self) -> None:
        super().__init__()

        self.experiment_name = "scpoli-ssl"

    def get_model(self, adata):
        return PrototypeScpoli(adata, self.latent_dims, self.num_prototypes)

    def train_step(self, model, train_loader, optimizer):
        return train_step(model, self.train_adata, train_loader, optimizer)

    def test_step(self, model, val_loader):
        return test_step(model, val_loader, self.val_adata)

    def get_ssl_representations(self):
        model = self.load_model()
        model.to(self.device)
        adata = self.dataset.adata
        data = generate_scpoli_dataset(adata, model.scpoli_model)
        return model.scpoli_model(data)


class LinearTrainer(ScpoliTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.num_prototypes = 32
        self.experiment_name = "linear-prot-scpoli-task_ratio10"

    def get_model(self, adata):
        head = nn.Linear(self.num_prototypes, self.dataset.num_classes, bias=False)
        return LinearPrototypeScpoli(adata, self.latent_dims, self.num_prototypes, head)

    def train_step(self, model, train_loader, optimizer):
        total_loss = PrototypeLoss()
        for scpoli_batch in train_loader:
            batch_loss = model(scpoli_batch)
            total_loss += batch_loss
            optimize_model(batch_loss, optimizer)
        total_loss.normalize(len(train_loader))
        return total_loss

    def test_step(self, model, test_loader):
        model.eval()
        test_loss = PrototypeLoss()
        with torch.inference_mode():
            for scpoli_batch in test_loader:
                test_loss += model(scpoli_batch)
        test_loss.normalize(len(test_loader))
        return test_loss


class BarlowTrainer(SslTrainer):
    def __init__(self) -> None:
        print("running barlow-scpoli")
        super().__init__()
        self.experiment_name = "barlow-scpoli"

    def get_model(self, adata):
        return BarlowScpoli(adata, self.latent_dims)


class ScpoliOriginal(ScpoliTrainer):

    def __init__(self) -> None:
        super().__init__()
        condition_key = "study"
        cell_type_key = "cell_type"
        self.scpoli_trainer = scPoli(
            adata=self.ref.adata,
            condition_keys=condition_key,
            cell_type_keys=cell_type_key,
            latent_dim=self.latent_dims,
            recon_loss="nb",
        )

    def get_model(self, adata):

        return self.scpoli_trainer.model

    def train(self):
        self.scpoli_trainer.train(
            n_epochs=100,
            pretraining_epochs=50,
        )
        path = self.get_model_path()
        utils.save_model(self.scpoli_trainer.model, path)
        
    def get_representation(self, model, adata):
        return self.scpoli_trainer.get_latent(adata, mean=True)
    
    def get_model_name(self):
        return f'original-scpoli-latent{self.latent_dims}.pth'