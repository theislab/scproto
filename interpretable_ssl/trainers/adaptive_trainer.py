from interpretable_ssl.trainers.scpoli_trainer import *
from torch.utils.data import random_split
from logging import getLogger
from interpretable_ssl.utils import log_time

logger = getLogger()


class AdoptiveTrainer(ScpoliTrainer):
    # @log_time('adoptive trainer')
    def __init__(self, debug=False, dataset=None, ref_query = None, parser=None, **kwargs) -> None:
       
        super().__init__(debug, dataset, ref_query, parser, **kwargs)
        self.finetune_ds = None
        self.original_ref = None
        self.partial_ref = None
        self.finetuning = False
        self.transfer_learning_mode = False
       
    # split train dataset
    # pretrain model
    # finetune on small portion of the model
    def tune_nmb_crops(self, adata_list):
        pass

    def split_train_data(self, finetune_size=0.1):
        self.original_ref = self.ref
        self.partial_ref, self.finetune_ds = self.ref.split(finetune_size)
        self.tune_nmb_crops([self.partial_ref.adata, self.finetune_ds.adata])

    def finetune(self):
        pass



    def train_semi_supervised(self):
        self.split_train_data()
        self.ref = self.partial_ref
        self.setup()
        self.train()
        self.finetuning = True
        self.ref = self.finetune_ds
        self.finetune()
        # keep finetuning true for scib evaluation on the run

    def transfer_learning(self):
        # pretrain on one dataset
        # finetune on another dataset
        # make sure about namings

        # pretrain
        self.dataset = self.get_dataset(self.pretrain_dataset_id)
        self.ref, self.query = self.dataset.get_train_test()
        self.setup()
        self.train()

        # finetune
        self.finetuning = True
        self.dataset = self.get_dataset(self.finetune_dataset_id)
        self.ref, self.query = self.dataset.get_train_test()
        # self.setup()
        self.finetune()

    def train(self):
        pass

    def setup(self):
        pass

    def run(self):

        if self.training_type == "semi_supervised":
            self.train_semi_supervised()
        elif self.training_type == "transfer_learning":
            self.transfer_learning()
        else:
            self.train()

        self.plot_ref_umap()
        self.plot_query_umap()
        self.additional_plots()
        # moved to the end of train function so we have scib metrics for both pretrain and finetuned version
        # self.save_scib_metrics()
