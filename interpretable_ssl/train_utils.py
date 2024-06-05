from interpretable_ssl.loss_manager import PrototypeLoss
import torch
import wandb


# Function to log gradient norms
def log_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"grad_norm_{name}": param.grad.norm().item()})


def optimize_model(batch_loss: PrototypeLoss, optimizer, model):
    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    batch_loss.overal.backward()
    # log_gradients(model)

    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # or some other value

    # 5. Optimizer step
    optimizer.step()


# def optimize_ssl_model(x1, x2, model, optimizer, overal_loss):
#     # 1. Forward pass
#     # 2. Calculate loss
#     batch_loss = model(x1, x2)
#     overal_loss += batch_loss

#     # 3. Optimizer zero grad
#     optimizer.zero_grad()

#     # 4. Loss backward
#     batch_loss.overal.backward()

#     # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # or some other value

#     # 5. Optimizer step
#     optimizer.step()
#     return overal_loss


# TODO: rename to ssl_train_step. add and refactpor normal train_step
def train_step(model, data_loader, optimizer, data_loader_size=None):

    overal_loss = PrototypeLoss()

    for x1, x2 in data_loader:
        # x1, x2 = x1.to(device), x2.to(device)

        # if data_loader.dataset.dataset.multiple_augment_cnt:
        #     for x2_item in x2:
        #         optimize_model(x1, x2_item, model, optimizer, overal_loss)
        # else:
        batch_loss = model(x1, x2)
        overal_loss += batch_loss
        optimize_model(batch_loss, optimizer, model)

    if not data_loader_size:
        data_loader_size = len(data_loader)
    overal_loss.normalize(data_loader_size)
    return overal_loss


def test_step(model, data_loader, data_loader_size=None):

    test_loss = PrototypeLoss()
    model.eval()  # put model in eval mode

    # Turn on inference context manager
    with torch.inference_mode():
        for x1, x2 in data_loader:
            # x1, x2 = x1.to(device), x2.to(device)

            # 1. Forward pass
            # 2. Calculate loss
            batch_loss = model(x1, x2)
            test_loss += batch_loss
        if not data_loader_size:
            data_loader_size = len(data_loader)
        test_loss.normalize(data_loader_size)
    return test_loss
