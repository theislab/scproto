from interpretable_ssl.trainers.scpoli_trainer import *


def train_using_prot_cnt(prot_cnt):
    st = ScpoliTrainer()
    st.num_prototypes = prot_cnt
    st.latent_dims = 8
    train_loss, val_loss = st.train(50)
    return train_loss, val_loss

def train_using_diffrent_prototypes():
    wandb.init(
        # set the wandb project where this run will be logged
        project="interpretable-ssl",
        # track hyperparameters and run metadata
        config={
            "description": "compare model performance trained using diffrent number of prototypes"
        },
    )
    prot_cnts = [70, 80, 90, 128]
    for prot_cnt in prot_cnts:
        train_loss, val_loss = train_using_prot_cnt(prot_cnt)
        wandb.log({'train': train_loss, 'test': val_loss, 'prot_cnt': prot_cnt})