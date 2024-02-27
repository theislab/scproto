import utils
from pancras_data import *
from prototype_classifier import ProtClassifier
import torch.optim as optim
from pathlib import Path
import wandb
import time
from tqdm.auto import tqdm
import prototype_classifier

if __name__ == "__main__":
    # load data
    device = utils.get_device()
    batch_size = 64
    pancras_data = PancrasDataset(device)
    train_loader, test_loader = utils.get_train_test_loader(pancras_data, batch_size)

    # define model
    num_classes = 14
    num_prototypes, num_classes = 16, 14
    input_dim, hidden_dim, latent_dims = 4000, 64, 8
    model = ProtClassifier(
        num_prototypes=num_prototypes,
        num_classes=num_classes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dims=latent_dims,
    )

    # init training parameter and wandb
    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    model_name = "pancras-prototype-classifier"
    model_path = Path.home() / f"/models/{model_name}.pth"

    # init wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="interpretable-ssl",
        # track hyperparameters and run metadata
        config={
            "model": model_name,
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
        train_loss_dict = prototype_classifier.add_prefix_key(train_loss.__dict__, "train")

        test_loss = prototype_classifier.test_step(test_loader, model, device)
        test_loss_dict = prototype_classifier.add_prefix_key(test_loss.__dict__, "test")

        train_loss_dict.update(test_loss_dict)

        wandb.log(train_loss_dict)
        if test_loss.acc > best_test_acc:
            utils.save_model_checkpoint(model, optimizer, epoch, model_path)

    print(f"training took {time.time() - st} seconds")
