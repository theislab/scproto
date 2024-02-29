from pancras_data import PancrasDataset
import utils
from torch.utils.data import random_split
import torch
import wandb
from tqdm.auto import tqdm
import torch.optim as optim

def get_model_name():
    return 'pca-linear.pth'

def get_model_path():
    return utils.get_pancras_model_dir() + get_model_name()
def train_step(model, loss_fn, optimizer, X_train, y_train):
    # Forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def save_model_checkpoint(model, opt, epoch, acc, save_path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'acc': acc,
            }, save_path)

def calculate_acc_and_loss(model, X, y, loss_fn):
    y_pred = model(X)
    _, predicted = torch.max(y_pred, dim=1)
    accuracy = (predicted == y).float().mean()
    loss = loss_fn(y_pred, y)
    return accuracy, loss

if __name__ == "__main__":

    device = utils.get_device()
    # load data
    pancras = PancrasDataset(device, True)
    train, test = pancras.get_train_test()
    X_train, y_train = train.dataset.x, train.dataset.y
    X_test, y_test = test.dataset.x, test.dataset.y

    # define model
    # build model
    model = torch.nn.Sequential(torch.nn.Linear(8, 14, bias=False))
    model.to(device)

    # define train and test step : above

    # init training params and wandb
    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # init wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="interpretable-ssl",
        # track hyperparameters and run metadata
        config={
            "model": "pca-linear",
            "epochs": epochs,
            "device": device,
            "model path": "",
        },
    )
    
    # train loop
    best_acc = -1
    print('start training')
    save_model_path = utils.get_pancras_model_dir() + get_model_name()
    for epoch in tqdm(range(epochs)): 
        loss = train_step(model, loss_fn, optimizer, X_train, y_train)

        with torch.no_grad():
            train_acc, train_loss= calculate_acc_and_loss(model, X_train, y_train, loss_fn)
            test_acc, test_loss = calculate_acc_and_loss(model, X_test, y_test, loss_fn)

            if train_acc > best_acc:
                best_acc = train_acc
                save_model_checkpoint(model, optimizer, epoch, train_acc, save_model_path)

            wandb.log({'train_acc': train_acc,
                    'train_loss': train_loss, 
                    'test_acc': test_acc, 
                    'test_loss': test_loss})