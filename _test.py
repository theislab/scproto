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