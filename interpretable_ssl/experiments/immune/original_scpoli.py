from interpretable_ssl.datasets.immune import ImmuneDataset
from interpretable_ssl.trainers.scpoli_original import *
from itertools import combinations


def get_scpoli_immune_trainer():
    ds = ImmuneDataset()
    trainer = OriginalTrainer(ds)
    return trainer


def train_scpoli_using_original_code():
    ds = ImmuneDataset()
    trainer = OriginalTrainer(ds)
    trainer.train(100)


def scpoli_original_five_fold():
    print("running scpoli original 5 fold")
    ds = ImmuneDataset()
    trainer = OriginalTrainer(ds)
    trainer.train_kfold_cross_val(50)


def scpoli_ccross_val_immune():
    trainer = get_scpoli_immune_trainer()
    trainer.train_custom_cross_val(100)


def scpoli_immune_bs256():
    trainer = get_scpoli_immune_trainer()
    trainer.batch_size = 256
    trainer.train(50)


def scpoli_immune_bs(bs):
    trainer = get_scpoli_immune_trainer()
    trainer.batch_size = bs
    trainer.train(100)


def scpoli_ccross_val_immune_bs(bs):
    trainer = get_scpoli_immune_trainer()
    trainer.batch_size = bs
    trainer.train_custom_cross_val(100)


def scpoli_epoch(epochs):
    trainer = get_scpoli_immune_trainer()
    trainer.experiment_name += f"_e{epochs}"
    trainer.train(epochs)


def scpoli_epoch_bs(epochs, bs):
    trainer = get_scpoli_immune_trainer()
    trainer.batch_size = bs
    trainer.experiment_name += f"_e{epochs}"
    trainer.train(epochs)


def scpoli_train_fold_1():
    trainer = get_scpoli_immune_trainer()
    trainer.fold = 1
    trainer.batch_size = 512
    trainer.custom_cross_val = True
    study_ids = trainer.dataset.get_study_ids()
    num_studies = len(study_ids)

    # Generate all combinations of test sets with exactly two studies
    test_study_indices = list(combinations(range(num_studies), 2))[0]
    train_study_indices = [i for i in range(num_studies) if i not in test_study_indices]
    test_study_indices = list(test_study_indices)
    trainer.ref, trainer.query = trainer.dataset.get_fold_train_test(
        train_study_indices, test_study_indices
    )
    trainer.train(100)
