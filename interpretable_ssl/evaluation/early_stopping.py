import torch
import wandb

def train_step(engine, batch):
    model = engine.state.model
    optimizer = engine.state.optimizer
    criterion = engine.state.criterion
    encode_batch = engine.state.encode_batch
    pretrained_model = engine.state.pretrained_model

    model.train()
    optimizer.zero_grad()
    batch_X = encode_batch(pretrained_model, batch)
    outputs = model(batch_X)
    loss = criterion(outputs, batch["celltypes"].squeeze())
    loss.backward()
    optimizer.step()
    wandb.log({'epoch': engine.state.epoch, 'linear loss': loss.item()})
    return loss.item()

def validation_step(engine, batch):
    model = engine.state.model
    # criterion = engine.state.criterion
    encode_batch = engine.state.encode_batch
    pretrained_model = engine.state.pretrained_model

    model.eval()
    with torch.no_grad():
        batch_X = encode_batch(pretrained_model, batch)
        outputs = model(batch_X)
        # loss = criterion(outputs, batch["celltypes"].squeeze())
        return outputs, batch["celltypes"].squeeze()

def attach_early_stopping(evaluator, trainer, patience=5):
    from ignite.handlers import EarlyStopping
    from ignite.engine import Events

    def score_function(engine):
        return -engine.state.metrics['loss']

    early_stopping_handler = EarlyStopping(
        patience=patience,
        score_function=score_function,
        trainer=trainer
    )
    evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)
