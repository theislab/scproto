import interpretable_ssl.utils as utils
import torch.optim as optim
import wandb
from tqdm.auto import tqdm
import interpretable_ssl.models.prototype_classifier as prototype_classifier
from interpretable_ssl.models.prototype_classifier import ProtClassifier
from interpretable_ssl.dataset import SingleCellDataset
from torch.utils.data import DataLoader
from pathlib import Path

class Trainer:
    def __init__(self, partially_train_ratio=None) -> None:
        self.num_prototypes = 8
        self.hidden_dim, self.latent_dims = 64, 8
        self.batch_size = 64

        self.device = utils.get_device()
        self.dataset = self.get_dataset()
        self.input_dim = self.dataset.x_dim

        self.partially_train_ratio = partially_train_ratio
        self.description = None

    def get_dataset(self) -> SingleCellDataset:
        pass

    def get_model_name(self):
        base = f"num-prot-{self.num_prototypes}_hidden-{self.hidden_dim}_bs-{self.batch_size}"
        if self.partially_train_ratio:
            return f"{base}_train-ratio-{self.partially_train_ratio}.pth"
        return base + ".pth"

    def get_model(self):
        pass

    def get_model_path(self):
        name = self.get_model_name()
        save_dir = utils.get_model_dir() + f'/{self.dataset.name}/'
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
    
    def train_step(self, model, train_loader, optimizer):
        pass
    
    def test_step(self, model, test_loader):
        pass
    
    def train(self):

        # load data
        train_loader, test_loader = self.get_train_test_loader()

        # define model
        model = self.get_model()

        # init training parameter and wandb
        epochs = 100
        optimizer = optim.Adam(model.parameters(), lr=0.001)
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
                "epochs": epochs,
                "device": self.device,
                "model path": model_path,
                "dataset": self.dataset,
                "partially train ratio": self.partially_train_ratio,
                "train size": len(train_loader.dataset),
                "test size": len(test_loader.dataset),
                "batch size": self.batch_size,
                "description": self.description
            },
        )

        # train loop
        best_test_acc = 0
        print("start training")
        for epoch in tqdm(range(epochs)):
            train_loss = self.train_step(
                model, train_loader, optimizer
            )
            train_loss_dict = prototype_classifier.add_prefix_key(
                train_loss.get_dict(), "train"
            )

            test_loss = self.test_step(test_loader, model)
            test_loss_dict = prototype_classifier.add_prefix_key(
                test_loss.get_dict(), "test"
            )

            train_loss_dict.update(test_loss_dict)

            wandb.log(train_loss_dict)
            if test_loss.acc > best_test_acc:
                utils.save_model_checkpoint(model, optimizer, epoch, model_path)
