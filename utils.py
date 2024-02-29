import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_home():
    return "/home/icb/fatemehs.hashemig/"

def get_train_test_loader(dataset, batch_size):

    train, test = random_split(dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
    train_loader, test_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True
    ), DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
def save_model_checkpoint(model, opt, epoch, save_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
        },
        save_path,
    )
    
def get_pancras_model_dir():
    return get_home() + '/models/pancras/'