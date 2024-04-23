import interpretable_ssl.utils as utils
import torch.optim as optim
import wandb
from tqdm.auto import tqdm
import interpretable_ssl.models.prototype_classifier as prototype_classifier
from interpretable_ssl.models.prototype_classifier import ProtClassifier
from interpretable_ssl.dataset import SingleCellDataset
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from interpretable_ssl.models import autoencoder, prototype_barlow
import torch


class Trainer:
    def __init__(self, partially_train_ratio=None, self_supervised=False, epochs=100) -> None:
        self.num_prototypes = 8
        self.hidden_dim, self.latent_dims = 64, 8
        self.batch_size = 64

        self.device = utils.get_device()
        self.dataset = self.get_dataset()
        self.self_supervised = self_supervised
        self.dataset.self_supervised = self_supervised

        self.input_dim = self.dataset.x_dim

        self.partially_train_ratio = partially_train_ratio
        self.description = None
        self.epochs = epochs
    def load_model(self):
        model = self.get_model()
        path = self.get_model_path()
        model.load_state_dict(torch.load(path)["model_state_dict"])
        return model

    def get_dataset(self) -> SingleCellDataset:
        pass

    def get_model_name(self):
        base = f"num-prot-{self.num_prototypes}_hidden-{self.hidden_dim}_bs-{self.batch_size}"
        if self.self_supervised:
            base = f"ssl-{base}"
        if self.partially_train_ratio:
            return f"{base}_train-ratio-{self.partially_train_ratio}.pth"
        return base + ".pth"

    def get_model(self):
        if self.self_supervised:
            return self.get_self_supervised_model()
        return self.get_classification_model()

    def get_classification_model(self):
        num_classes = self.dataset.num_classes

        model = ProtClassifier(
            num_prototypes=self.num_prototypes,
            num_classes=num_classes,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dims=self.latent_dims,
        )
        return model

    def get_self_supervised_model(self):
        vae = autoencoder.VariationalAutoencoder(
            self.input_dim, self.hidden_dim, self.latent_dims
        )
        model = prototype_barlow.PrototypeBarlow(vae, self.num_prototypes)
        return model

    def get_model_path(self):
        name = self.get_model_name()
        save_dir = utils.get_model_dir() + f"/{self.dataset.name}/"
        Path.mkdir(Path(save_dir), exist_ok=True)
        return save_dir + name

    def get_train_test_loader(self):
        train, test = self.dataset.get_train_test()
        if self.partially_train_ratio:
            train = utils.sample_dataset(train, self.partially_train_ratio)
            test = utils.sample_dataset(test, self.partially_train_ratio)
        print("train dataset size: ", len(train))
        train_loader, test_loader = DataLoader(
            train, batch_size=self.batch_size, shuffle=True
        ), DataLoader(test, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def ssl_train_step(self, model, train_loader, optimizer):
        return prototype_barlow.train_step(model, train_loader, optimizer, self.device)

    def ssl_test_step(self, model, test_loader):
        return prototype_barlow.test_step(model, test_loader, self.device)

    def calssification_train_step(self, model, train_loader, optimizer):
        return prototype_classifier.train_step(
            model, train_loader, optimizer, self.device
        )

    def classification_test_step(self, model, test_loader):
        return prototype_classifier.test_step(test_loader, model, self.device)

    def train_step(self, model, train_loader, optimizer):
        if self.self_supervised:
            return self.ssl_train_step(model, train_loader, optimizer)
        else:
            return self.calssification_train_step(model, train_loader, optimizer)

    def test_step(self, model, test_loader):
        if self.self_supervised:
            return self.ssl_test_step(model, test_loader)
        else:
            return self.classification_test_step(model, test_loader)

    def train(self):

        # load data
        train_loader, test_loader = self.get_train_test_loader()

        # define model
        model = self.get_model()

        # init training parameter and wandb
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        model_path = self.get_model_path()

        # init wandb
        wandb.init(
            # set the wandb project where this run will be logged
            project="interpretable-ssl",
            # track hyperparameters and run metadata
            config={
                "num_prototypes": self.num_prototypes,
                "hidden dim": self.hidden_dim,
                "latent_dims": self.latent_dims,
                "epochs": self.epochs,
                "device": self.device,
                "model path": model_path,
                "dataset": self.dataset,
                "partially train ratio": self.partially_train_ratio,
                "train size": len(train_loader.dataset),
                "test size": len(test_loader.dataset),
                "batch size": self.batch_size,
                "description": self.description,
                "self-supervised": self.self_supervised,
            },
        )

        # train loop
        best_test_loss = sys.maxsize
        print("start training")
        for epoch in tqdm(range(self.epochs)):
            train_loss = self.train_step(model, train_loader, optimizer)
            train_loss_dict = prototype_classifier.add_prefix_key(
                train_loss.__dict__, "train"
            )

            test_loss = self.test_step(model, test_loader)
            test_loss_dict = prototype_classifier.add_prefix_key(
                test_loss.__dict__, "test"
            )

            train_loss_dict.update(test_loss_dict)

            wandb.log(train_loss_dict)
            if test_loss.loss < best_test_loss:
                utils.save_model_checkpoint(model, optimizer, epoch, model_path)

    def get_ssl_representations(self):
        model = self.load_model()
        model.to(self.device)
        x = self.dataset.adata.X.toarray()
        x = torch.tensor(x, device=self.device)
        latent, _, prot_dist = model.prototype_model(x)
        projections = model.barlow_model.projector(prot_dist)
        return (
            utils.tensor_to_numpy(latent),
            utils.tensor_to_numpy(prot_dist),
            utils.tensor_to_numpy(projections),
        )
