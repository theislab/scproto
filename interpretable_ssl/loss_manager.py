import torch
import torch.nn.functional as F

def repulsion_loss(prototypes):
    """
    Calculate repulsion loss to ensure prototypes are spread out.
    """
    dist_matrix = torch.cdist(prototypes, prototypes, p=2)
    # Invert distances to get repulsion force (1/distance)
    inv_dist_matrix = 1.0 / (dist_matrix + 1e-5)
    repulsion = torch.sum(torch.triu(inv_dist_matrix, diagonal=1))
    return repulsion


class PrototypeLoss:
    def __init__(self) -> None:
        # print('------0.01 vae_reg---------')
        self.vae, self.interpretability = 0, 0
        self.overal = 0
        self.task = 0
        self.vae_reg = 0.01
        self.repulsion_loss, self.repulsion_alpha = 0, 0.005
        self.fixed_values = ['vae_reg', 'fixed_values', 'repulsion_alpha']
        # self.to_norm_loss_keys = ['vae', 'task', 'interpretability']
        # self.max_values = {self.__dict__[key] for key in self.to_norm_loss_keys}
    def calculate_overal(self, vae, interpretability, task=0):
        self.vae = vae
        self.interpretability = interpretability
        self.task = task
        self.overal = self.interpretability + self.vae_reg * self.vae + self.task
        # self.overal = self.vae

    def __add__(self, prot_loss):
        # new_loss = PrototypeLoss()
        for key in self.__dict__:
            if key in self.fixed_values:
                continue
            self.__dict__[key] = prot_loss.__dict__[key] + self.__dict__[key]
        return self

    def normalize(self, data_loader_size):
        for key, val in self.__dict__.items():
            if key in self.fixed_values:
                continue
            self.__dict__[key] = val / data_loader_size
            
        return self

    def set_task_loss(self, task_loss, task_ratio=1):
        self.task = task_loss
        self.overal += self.task * task_ratio
        
    def set_repulsion(self, prototypes):
        self.repulsion_loss = int(repulsion_loss(prototypes))
        self.overal += self.repulsion_alpha * self.repulsion_loss
        