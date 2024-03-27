# load data
# load model
# calculate latent
# process data to work
# calculate metric

from interpretable_ssl.pancras.train.train import PancrasTrainer
from torch.utils.data import DataLoader
import scib

def calculate_latent(model, loader):
    latents = []
    for x, y in loader:
        latent = model(x)
        latents.append(latent)
    return 
def main():
    
    trainer = PancrasTrainer(split_study=True)
    model = trainer.get_model()
    train_loader, test_loader = trainer.get_train_test_loader()
    
if __name__ == "__main__":
    scib.metrics.metrics()