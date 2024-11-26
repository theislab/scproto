from interpretable_ssl.trainers.scpoli_trainer import *
from torch.utils.data import random_split
from logging import getLogger
from interpretable_ssl.utils import log_time
from interpretable_ssl.trainers.cvae_trainer import CvaeTrainer
logger = getLogger()


class AdoptiveTrainer(ScpoliTrainer):
    # @log_time('adoptive trainer')
    def __init__(
        self, debug=False, dataset=None, ref_query=None, parser=None, **kwargs
    ) -> None:

        super().__init__(debug, dataset, ref_query, parser, **kwargs)
        self.finetune_ds = None
        self.original_ref = self.ref
        self.partial_ref = None
        self.finetuning = False
        self.transfer_learning_mode = False

    # split train dataset
    # pretrain model
    # finetune on small portion of the model
    def tune_nmb_crops(self, adata_list):
        pass

    def split_train_data(self, finetune_size=0.1):
        self.partial_ref, self.finetune_ds = self.original_ref.split(finetune_size)
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

    def train_partial_supervised(self):
        self.split_train_data()
        self.finetuning = True
        self.ref = self.finetune_ds
        self.finetune()

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

    def train_fully_supervised(self):
        pass

    def pretrain_decodabale_proto(self):
        cvae_trainer = CvaeTrainer(self.model, self.batch_size, self.debug, self.dataset, (self.ref, self.query), original_ref=self.original_ref)
        cvae_trainer.train(self.cvae_epochs)
        
    def pretrain_encoder(self):
        if self.decodable_prototypes == 0:
            self.get_scpoli(self.model, False).train(
                n_epochs=self.cvae_epochs,
                pretraining_epochs=self.cvae_epochs,
                eta=5,
            )
        else:
            self.pretrain_decodabale_proto()

    def train_pretrain_encoder(self):
        if self.fine_tuning_epochs > 0:
            self.split_train_data()
            self.ref = self.partial_ref
        self.setup()
        self.pretrain_encoder()
        self.init_prototypes()

        # pretrain with swav
        self.train()

        if self.fine_tuning_epochs > 0:
            self.prot_decoding_loss_scaler = 0
            self.finetuning = True
            self.ref = self.finetune_ds
            self.finetune()

    def train(self):
        pass

    def setup(self):
        pass

    def get_umap_path(self, data_part="ref"):
        img_name = f"/{data_part}-umap"
        if self.finetuning:
            img_name = f"{img_name}_finetuned"
        umap_paths = [self.get_dump_path() + f"/{img_name}.png"]
        if self.save_temp_res == 1:
            umap_paths.append(self.get_temp_res_path() + f"/{img_name}.png")
        return umap_paths

    def load_adopt(self):
        model = self.load_model()
        self.adapt_ref_model(model, self.finetune_ds.adata)
        return model

    def run(self):
        if not self.debug:
            self.init_wandb(self.dump_path)
        if self.training_type == "semi_supervised":
            self.train_semi_supervised()
        elif self.training_type == "transfer_learning":
            self.transfer_learning()
        elif self.training_type == "partial_supervised":
            self.train_partial_supervised()
        elif self.training_type == "fully_supervised":
            self.train_fully_supervised()
        elif self.training_type == "pretrain_encoder":
            self.train_pretrain_encoder()
        else:
            self.train()

        self.ref = self.original_ref
        # self.plot_ref_umap(name_postfix="with-model_all", model=self.model)
        # self.plot_ref_umap(name_postfix="load-adopt_all", model=self.load_adopt())
        self.plot_ref_umap(name_postfix="all", model=self.model)
        self.plot_query_umap()
        self.additional_plots()
        # moved to the end of train function so we have scib metrics for both pretrain and finetuned version
        # self.save_scib_metrics()
