import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import apex
from apex.parallel.LARC import LARC

from swav.src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
)

from interpretable_ssl.trainers.scpoli_trainer import ScpoliTrainer
from interpretable_ssl.augmenters.adata_augmenter import *
from scarches.models.scpoli import scPoli
import scarches.trainers.scpoli._utils as scpoli_utils
from interpretable_ssl.models.swav import SwavModel
import wandb
import multiprocessing as mp
from interpretable_ssl.evaluation.visualization import *
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from interpretable_ssl.configs.defaults import get_swav_defaults
from interpretable_ssl.configs.parsers import add_swav_parser_args

logger = getLogger()

class SwAV(ScpoliTrainer):
    def __init__(self, parser=None, **kwargs):
        super().__init__(None, **kwargs)

        if parser is not None:
            self.args = add_swav_parser_args(parser).parse_args()
        else:
            defaults = get_swav_defaults()
            defaults.update(kwargs)
            self.args = argparse.Namespace(**defaults)
        
        self.experiment_name = self.args.experiment_name
        self.set_data_parameters(self.args)
        self.set_swav_parameters(self.args)
        self.set_optimization_parameters(self.args)
        self.set_distribution_parameters(self.args)
        self.set_other_parameters(self.args)
        self.set_optional_fields(self.args)

        self.create_dump_path()
        self.seed = self.args.seed
        
    def set_data_parameters(self, args):
        self.nmb_crops = args.nmb_crops
        self.augmentation_type = args.augmentation_type
        self.size_crops = args.size_crops
        self.min_scale_crops = args.min_scale_crops
        self.max_scale_crops = args.max_scale_crops

    def set_swav_parameters(self, args):
        self.crops_for_assign = args.crops_for_assign
        self.temperature = args.temperature
        self.epsilon = args.epsilon
        self.sinkhorn_iterations = args.sinkhorn_iterations
        self.latent_dims = args.latent_dims
        self.feat_dim = self.latent_dims
        self.num_prototypes = args.num_prototypes
        self.nmb_prototypes = self.num_prototypes
        self.queue_length = args.queue_length
        self.epoch_queue_starts = args.epoch_queue_starts

    def set_optimization_parameters(self, args):
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.base_lr = args.base_lr
        self.final_lr = args.final_lr
        self.freeze_prototypes_niters = args.freeze_prototypes_niters
        self.wd = args.wd
        self.warmup_epochs = args.warmup_epochs
        self.start_warmup = args.start_warmup
        self.cvae_loss_scaler = args.cvae_loss_scaler
        self.prot_decoding_loss_scaler = args.prot_decoding_loss_scaler

    def set_distribution_parameters(self, args):
        self.dist_url = args.dist_url
        self.world_size = args.world_size
        self.rank = args.rank
        self.local_rank = args.local_rank

    def set_other_parameters(self, args):
        self.workers = args.workers
        self.checkpoint_freq = args.checkpoint_freq
        self.use_fp16 = args.use_fp16
        self.sync_bn = args.sync_bn
        self.syncbn_process_group_size = args.syncbn_process_group_size
        self.dump_name_version = args.dump_name_version

    def set_optional_fields(self, args):
        self.model = args.model
        self.optimizer = args.optimizer
        self.lr_schedule = args.lr_schedule
        self.queue = args.queue
        self.train_loader = args.train_loader
        self.training_stats = args.training_stats
        self.condition_key = args.condition_key
        self.cell_type_key = args.cell_type_key
        self.device = args.device
        self.debug = args.debug
        self.freezable_prototypes = args.freezable_prototypes

    def create_dump_path(self):
        self.dump_path = self.get_dump_path()
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)

    def get_dump_path(self):
        dump_path, _ = os.path.splitext(self.get_model_path())
        if self.dump_name_version != 1:
            dump_path = f"{dump_path}_aug{self.nmb_crops[0]}_latent{self.latent_dims}"
        if self.dump_name_version > 2:
            dump_path = f"{dump_path}_aug-type-{self.augmentation_type}"
        return dump_path

    def setup(self):
        fix_random_seeds(self.seed)
        logger, self.training_stats = initialize_exp(self, "epoch", "loss")
        self.init_scpoli()
        self.build_data()
        self.build_model()
        self.build_optimizer()
        self.init_mixed_precision()
        self.load_checkpoint()
        if not self.debug:
            self.init_wandb(self.dump_path, len(self.ref.adata), 0)

    def init_scpoli(self):

        self.scpoli_ = scPoli(
            adata=self.ref.adata,
            condition_keys=self.condition_key,
            cell_type_keys=self.cell_type_key,
            latent_dim=self.latent_dims,
            recon_loss="nb",
        )

    def build_data(self):

        # train, val = self.split_train_test(self.ref)
        # self.train_adata, self.val_adata = train, val
        train = self.ref.adata
        # why nmb_crops is a list? i used fisrt element but not change it in case needed in furure
        model = self.scpoli_.model
        self.train_ds = MultiCropsDataset(
            train,
            self.nmb_crops[0],
            self.augmentation_type,
            k_neighbors=10,
            condition_keys=[self.condition_key],
            cell_type_keys=[self.cell_type_key],
            condition_encoders=model.condition_encoders,
            conditions_combined_encoder=model.conditions_combined_encoder,
            cell_type_encoder=model.cell_type_encoder,
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=scpoli_utils.custom_collate,
        )
        logger.info(f"Building data done with {len(self.train_ds)} images loaded.")

    def get_model(self):
        return SwavModel(self.scpoli_.model, self.latent_dims, self.num_prototypes)

    def load_model(self):
        model = self.get_model()
        checkpoint_path = os.path.join(self.dump_path, "checkpoint.pth.tar")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.to("cuda")
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # model, _ = apex.amp.initialize(model, self.optimizer, opt_level="O1")
        self.scpoli_.model = model.scpoli_model
        return model

    def get_embeddings(self, adata):
        self.scpoli_.model.eval()
        with torch.no_grad():
            embeddings = self.scpoli_.get_latent(adata, mean=True)
        return embeddings

    def calculate_ref_embeddings(self, model):
        embeddings = []
        outputs = []
        total_cvae_loss = 0.0
        total_prot_decoding_loss = 0.0
        num_batches = 0
        scpoli_model = self.scpoli_.model
        train_ds = MultiConditionAnnotatedDataset(
            self.ref.adata,
            condition_keys=[self.condition_key],
            cell_type_keys=[self.cell_type_key],
            condition_encoders=scpoli_model.condition_encoders,
            conditions_combined_encoder=scpoli_model.conditions_combined_encoder,
            cell_type_encoder=scpoli_model.cell_type_encoder,
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            collate_fn=scpoli_utils.custom_collate,
        )
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient calculation
            for inputs in self.train_loader:
                inputs = self.move_input_on_device(inputs)
                embedding, output, cvae_loss, prot_decoding_loss = model(inputs)
                embeddings.append(embedding)
                outputs.append(output)
                total_cvae_loss += cvae_loss.item()
                total_prot_decoding_loss += prot_decoding_loss.item()
                num_batches += 1

        avg_cvae_loss = total_cvae_loss / num_batches
        avg_prot_decoding_loss = total_prot_decoding_loss / num_batches

        embeddings = torch.cat(embeddings)
        outputs = torch.cat(outputs)

        return embeddings, outputs, avg_cvae_loss, avg_prot_decoding_loss

    def build_model(self):
        self.model = self.get_model()

        self.model = self.model.cuda()
        logger.info(self.model)
        logger.info("Building model done.")

    def build_optimizer(self):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=0.9,
            weight_decay=self.wd,
        )
        self.optimizer = LARC(
            optimizer=self.optimizer, trust_coefficient=0.001, clip=False
        )

        warmup_lr_schedule = np.linspace(
            self.start_warmup, self.base_lr, len(self.train_loader) * self.warmup_epochs
        )
        iters = np.arange(len(self.train_loader) * (self.epochs - self.warmup_epochs))
        cosine_lr_schedule = np.array(
            [
                self.final_lr
                + 0.5
                * (self.base_lr - self.final_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * t
                        / (len(self.train_loader) * (self.epochs - self.warmup_epochs))
                    )
                )
                for t in iters
            ]
        )
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

        logger.info("Building optimizer done.")

    def init_mixed_precision(self):
        if self.use_fp16:
            self.model, self.optimizer = apex.amp.initialize(
                self.model, self.optimizer, opt_level="O1"
            )
            logger.info("Initializing mixed precision done.")

    def load_checkpoint(self):
        to_restore = {"epoch": 0}
        restart_from_checkpoint(
            os.path.join(self.dump_path, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=self.model,
            optimizer=self.optimizer,
            amp=apex.amp,
        )
        self.start_epoch = to_restore["epoch"]

    def save_checkpoint(self, epoch):
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_fp16:
            save_dict["amp"] = apex.amp.state_dict()
        torch.save(save_dict, os.path.join(self.dump_path, "checkpoint.pth.tar"))
        if epoch % self.checkpoint_freq == 0 or epoch == self.epochs - 1:
            shutil.copyfile(
                os.path.join(self.dump_path, "checkpoint.pth.tar"),
                os.path.join(self.dump_path, f"ckp-{epoch}.pth"),
            )

    def log_wandb_loss(self, scores):
        _, avg_loss, cvae_loss, swav_loss, prot_loss = scores
        wandb.log(
            {
                "swav": swav_loss,
                "cvae": cvae_loss,
                "loss": avg_loss,
                "prot loss": prot_loss,
            }
        )

    def run(self):
        cudnn.benchmark = True
        for epoch in range(self.start_epoch, self.epochs):
            logger.info(f"============ Starting epoch {epoch} ... ============")
            if (
                self.queue_length > 0
                and epoch >= self.epoch_queue_starts
                and self.queue is None
            ):
                self.queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.queue_length,
                    self.feat_dim,
                ).cuda()

            scores, self.queue = self.train_one_epoch(epoch)
            # self.training_stats.update(scores)
            self.log_wandb_loss(scores)
            self.save_checkpoint(epoch)

    def move_input_on_device(self, inputs):
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def train_one_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        cvae_losses = AverageMeter()
        swav_losses = AverageMeter()
        prot_decoding_losses = AverageMeter()

        self.model.train()
        use_the_queue = False

        end = time.time()
        for iteration, inputs in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            self.update_learning_rate(epoch, iteration)

            with torch.no_grad():
                w = self.model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                self.model.prototypes.weight.copy_(w)

            inputs = self.move_input_on_device(inputs)
            inputs = reshape_and_reorder_dict(inputs)
            embedding, output, cvae_loss, prot_decoding_loss = self.model(inputs)
            embedding = embedding.detach()

            bs = self.batch_size

            swav_loss = self.compute_swav_loss(embedding, output, bs, use_the_queue)
            loss = swav_loss + cvae_loss * self.cvae_loss_scaler + prot_decoding_loss * self.prot_decoding_loss_scaler
            self.optimizer.zero_grad()

            if self.use_fp16:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.freezable_prototypes:
                if (
                    epoch * len(self.train_loader) + iteration
                    < self.freeze_prototypes_niters
                ):
                    for name, p in self.model.named_parameters():
                        if "prototypes" in name:
                            p.grad = None

            self.optimizer.step()

            losses.update(loss.item(), inputs["x"].size(0))
            cvae_losses.update(cvae_loss.item(), inputs["x"].size(0))
            swav_losses.update(swav_loss.item(), inputs["x"].size(0))
            prot_decoding_losses.update(prot_decoding_loss.item(), inputs["x"].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if iteration % 50 == 0:
                logger.info(
                    f"Epoch: [{epoch}][{iteration}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Lr: {self.optimizer.param_groups[0]['lr']:.4f}"
                )

        return (
            epoch,
            losses.avg,
            cvae_losses.avg,
            swav_losses.avg,
            prot_decoding_losses.avg,
        ), self.queue

    def update_learning_rate(self, epoch, iteration):
        iteration = epoch * len(self.train_loader) + iteration
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[iteration]

    def compute_swav_loss(self, embedding, output, bs, use_the_queue):
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id : bs * (crop_id + 1)].detach()

                if self.queue is not None:
                    if use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        use_the_queue = True
                        queue_output = torch.mm(
                            self.queue[i], self.model.module.prototypes.weight.t()
                        )
                        out = torch.cat((queue_output, out))

                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs : (crop_id + 1) * bs]

                q = self.distributed_sinkhorn(out)[-bs:]

            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = output[bs * v : bs * (v + 1)] / self.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)
        return loss

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()

    def calculate_umaps(self, trained_model=True):

        if trained_model:
            model = self.load_model()
        else:
            model = self.get_model()
            model.to("cuda")
        embeddings, outputs, _, _ = self.calculate_ref_embeddings(model)
        prototypes = model.prototypes.weight
        prototypes = prototypes.detach().cpu()
        outputs = outputs.detach().cpu()
        embeddings = embeddings.detach().cpu()

        self.cell_types, self.study_labels = (
            self.ref.adata.obs.cell_type,
            self.ref.adata.obs.study,
        )
        self.embedding_umap, self.prototype_umap = calculate_umap(
            embeddings, prototypes, 'cosine'
        )
        self.outputs_umap, _ = calculate_umap(outputs, metric='cosine')

    def plot_ref_prot_umap(self):
        plot_umap(
            self.embedding_umap, self.prototype_umap, self.cell_types, self.study_labels
        )

    def plot_ref_mapping_umap(self):
        plot_umap(self.outputs_umap, None, self.cell_types, self.study_labels)

    def plot_by_augmentation(self, n_samples, n_augmentations):
        model = self.load_model()
        # Initialize the dataset with the entire adata
        self.train_ds.n_augmentations = n_augmentations

        # Sample indices from the entire dataset
        indices = np.random.choice(len(self.ref.adata), n_samples, replace=False)
        indices = [int(idx) for idx in indices]

        # Create a DataLoader for the sampled subset
        subset_dataset = Subset(self.train_ds, indices)
        dataloader = DataLoader(
            subset_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Initialize lists to store embeddings and labels
        all_embeddings = []
        all_labels = []

        for i, inputs in enumerate(dataloader):

            # Move inputs to device
            inputs = self.move_input_on_device(inputs)
            batch_size = inputs["x"].shape[0]
            labels = np.repeat(np.arange(batch_size), n_augmentations)
            labels = labels.reshape(batch_size, n_augmentations)
            labels = torch.tensor(labels)

            # Reshape and reorder the inputs
            inputs = reshape_and_reorder_dict(inputs)
            labels = reshape_and_reorde_tensor(labels)

            # Calculate embeddings
            with torch.no_grad():
                embeddings, _, _, _ = model(inputs)
                embeddings = embeddings.detach().cpu().numpy()

            # Generate labels for the augmentations
            batch_size = inputs["x"].shape[0] // n_augmentations

            # Append embeddings and labels to the lists
            all_embeddings.append(embeddings)
            all_labels.append(labels)

        # Concatenate all embeddings and labels
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        cell_umap, prototype_umap = calculate_umap(all_embeddings)
        plot_umap(
            cell_umap,
            prototype_umap,
            inputs["celltypes"].reshape(n_samples * n_augmentations).cpu(),
            all_labels,
        )


if __name__ == "__main__":
    swav = SwAV()
    swav.setup()
    swav.run()


# Example usage
# To run with command line arguments:
# python script.py --dataset some_dataset --dump_name_version 4 --nmb_crops 10 12 --augmentation_type knn --epochs 500
