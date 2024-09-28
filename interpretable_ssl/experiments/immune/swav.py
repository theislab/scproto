from interpretable_ssl.datasets.immune import ImmuneDataset
from interpretable_ssl.trainers.swav import *

def train_default_swav():
    dataset = ImmuneDataset()
    swav = SwAV(dataset)
    swav.setup()
    swav.run()