import utils
from interpretable_ssl.pancras.data import *
from interpretable_ssl.models.prototype_classifier import ProtClassifier
import torch.optim as optim
from pathlib import Path
import wandb
import time
from tqdm.auto import tqdm
import interpretable_ssl.models.prototype_classifier as prototype_classifier


def get_model_name(num_prototypes):
    return f"num_prot_{num_prototypes}.pth"


def get_model_path(num_prototypes):
    model_name = get_model_name(num_prototypes)
    return utils.get_pancras_model_dir() + model_name


num_prototypes = 8
input_dim, hidden_dim, latent_dims = 4000, 64, 8


def get_model():
    num_classes = 14

    model = ProtClassifier(
        num_prototypes=num_prototypes,
        num_classes=num_classes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dims=latent_dims,
    )
    return model


def main():
    # load data
    device = utils.get_device()
    batch_size = 64
    pdata = PancrasDataset(device)
    train_loader, test_loader = utils.get_train_test_loader(pdata, batch_size)

    # define model
    model = get_model()

    # init training parameter and wandb
    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    model_path = get_model_path(num_prototypes)

    # init wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="interpretable-ssl",
        # track hyperparameters and run metadata
        config={
            "num_prototypes": num_prototypes,
            "hidden dim": hidden_dim,
            "latent_dims": latent_dims,
            "epochs": epochs,
            "device": device,
            "model path": model_path,
        },
    )

    # train loop
    best_test_acc = 0
    print("start training")
    st = time.time()
    for epoch in tqdm(range(epochs)):
        train_loss = prototype_classifier.train_step(
            model, train_loader, optimizer, device
        )
        train_loss_dict = prototype_classifier.add_prefix_key(
            train_loss.__dict__, "train"
        )

        test_loss = prototype_classifier.test_step(test_loader, model, device)
        test_loss_dict = prototype_classifier.add_prefix_key(test_loss.__dict__, "test")

        train_loss_dict.update(test_loss_dict)

        wandb.log(train_loss_dict)
        if test_loss.acc > best_test_acc:
            utils.save_model_checkpoint(model, optimizer, epoch, model_path)

    print(f"training took {time.time() - st} seconds")
