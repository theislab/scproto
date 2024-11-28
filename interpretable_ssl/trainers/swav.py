import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
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

from interpretable_ssl.trainers.adaptive_trainer import AdoptiveTrainer
from interpretable_ssl.augmenters.adata_augmenter import *
from scarches.models.scpoli import scPoli
import scarches.trainers.scpoli._utils as scpoli_utils
from interpretable_ssl.models.swav import *
import wandb
import multiprocessing as mp
from interpretable_ssl.evaluation.visualization import *
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from interpretable_ssl.configs.defaults import *
import sys
from interpretable_ssl.utils import log_time

from interpretable_ssl.evaluation.prototype_metrics import *

logger = getLogger()


class SwAV(AdoptiveTrainer):

    # @log_time('swav')
    def __init__(
        self, debug=False, dataset=None, ref_query=None, parser=None, **kwargs
    ):

        self.is_swav = 1
        super().__init__(debug, dataset, ref_query, parser, **kwargs)
        self.nmb_prototypes = self.num_prototypes
        self.use_projector_out = False
        # would be defferent when trying to finetune, keep original aug type for model path
        self.train_augmentation = self.augmentation_type
        # print(self.temperature)
        # self.set_experiment_name()

    def setup(self):
        fix_random_seeds(self.seed)
        self.dump_path = self.get_dump_path()
        # self.create_dump_path()
        logger, self.training_stats = initialize_exp(self, "epoch", "loss")
        # self.init_scpoli()
        self.build_model()

        self.build_data()

        self.build_optimizer()
        self.init_mixed_precision()
        self.load_checkpoint()

    def build_data(self):

        # train, val = self.split_train_test(self.ref)
        # self.train_adata, self.val_adata = train, val

        # why nmb_crops is a list? i used fisrt element but not change it in case needed in furure
        # model = self.scpoli_.model
        scpoli_encoder = self.model.scpoli_encoder
        self.train_ds = MultiCropsDataset(
            self.ref.adata,
            self.ref.original_idx,
            self.nmb_crops[0],
            self.train_augmentation,
            k_neighbors=self.k_neighbors,
            longest_path=self.longest_path,
            dimensionality_reduction=self.dimensionality_reduction,
            condition_keys=[self.condition_key],
            # cell_type_keys=[self.cell_type_key],
            condition_encoders=scpoli_encoder.condition_encoders,
            conditions_combined_encoder=scpoli_encoder.conditions_combined_encoder,
            # cell_type_encoder=model.cell_type_encoder,
        )

        self.train_loader = self.get_data_laoder()
        logger.info(f"Building data done with {len(self.train_ds)} samples loaded.")

    def get_data_laoder(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=scpoli_utils.custom_collate,
            shuffle=True,
        )

    def get_model(self):
        # if self.model_version == 1:
        # model = SwavBase(self.scpoli_.model, self.latent_dims, self.num_prototypes)
        # return model
        # else:
        if self.decodable_prototypes == 1:
            return SwAVDecodableProto(
                self.latent_dims, self.num_prototypes, self.ref.adata
            )
        else:
            return SwAVModel(self.latent_dims, self.num_prototypes, self.ref.adata)

    def get_model_path(self):
        return os.path.join(self.get_dump_path(), self.get_checkpoint_file())

    def load_model(self):
        model = self.get_model()
        checkpoint_path = self.get_model_path()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)

        # self.optimizer.load_state_dict(checkpoint["optimizer"])
        # model, _ = apex.amp.initialize(model, self.optimizer, opt_level="O1")

        # model = apex.amp.initialize(model, opt_level="O1")

        # self.model = model
        return model

    def init_prototypes(self):
        if self.prot_init == "kmeans" and self.decodable_prototypes == 0:
            logger.info("initalizing prototypes using kmeans")
            embeddings = self.encode_ref(self.model)
            self.model.init_prototypes_kmeans(embeddings, self.nmb_prototypes)

    def build_model(self):
        self.model = self.get_model()
        self.model = self.model.cuda()
        self.init_prototypes()
        logger.info(self.model)
        logger.info(f"Building model done. with prot init {self.prot_init}")

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
        iters = np.arange(
            len(self.train_loader) * (self.pretraining_epochs - self.warmup_epochs)
        )
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
                        / (
                            len(self.train_loader)
                            * (self.pretraining_epochs - self.warmup_epochs)
                        )
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
        else:
            logger.info("no mixed precision")

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

    def get_checkpoint_file(self):
        if self.finetuning:
            checkpoint_file = "finetuned-checkpoint.pth.tar"
        elif self.training_type == "semi_supervised":
            checkpoint_file = "semi-pretrain-checkpoint.pth.tar"
        else:
            checkpoint_file = "checkpoint.pth.tar"
        return checkpoint_file

    def save_checkpoint(self, epoch):
        if self.debug:
            return
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_fp16:
            save_dict["amp"] = apex.amp.state_dict()

        checkpoint_file = self.get_checkpoint_file()
        torch.save(save_dict, os.path.join(self.dump_path, checkpoint_file))
        if epoch % self.checkpoint_freq == 0 or epoch == self.pretraining_epochs - 1:
            shutil.copyfile(
                os.path.join(self.dump_path, checkpoint_file),
                os.path.join(self.dump_path, f"ckp-{epoch}.pth"),
            )

    def log_wandb_loss(self, scores, epoch):
        (
            _,
            avg_loss,
            cvae_loss,
            swav_loss,
            propagation_loss,
            prot_emb_sim_loss,
            num_matches,
            prob_entropy,
            p_entropy,
        ) = scores
        log_dict = {
            "epoch": epoch,
            "swav": swav_loss,
            "cvae": cvae_loss,
            "loss": avg_loss,
            "propagation loss": propagation_loss,
            "prot_emb_sim_loss": prot_emb_sim_loss,
            "num matches": num_matches,
            "match ratio": num_matches / self.batch_size,
            "clustering counts entropy": prob_entropy,
            "prototype distance": self.model.prototypes_avg_distance(),
            "p_t entropy": p_entropy,
        }
        if epoch % 5 == 0:
            log_dict = log_dict | self.calculate_prototype_metrics()
        if not self.debug:
            wandb.log(log_dict)
        else:
            print(log_dict)

    def train(self, epochs=None):
        self.create_dump_path()
        if self.check_scib_metrics_exist():
            self.model = self.load_model()
            return

        cudnn.benchmark = True
        if epochs is None:
            epochs = self.pretraining_epochs
        for epoch in range(self.start_epoch, epochs):
            logger.info(f"============ Starting epoch {epoch}============")

            if epoch % self.umap_checkpoint_freq == 0:
                # self.plot_ref_umap(name_postfix=f'e{epoch}', model=self.model)
                self.plot_umap(self.model, self.original_ref.adata, f"ref-e{epoch}")

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
            # self.scpoli_.model = self.model.scpoli_model
            # self.training_stats.update(scores)
            self.log_wandb_loss(scores, epoch)
            self.save_checkpoint(epoch)

        # if self.train_decoder:
        #     self.only_decoder_train()
        self.plot_umap(self.model, self.original_ref.adata, "ref")
        self.plot_query_umap()
        try:
            self.save_metrics()
        except Exception as e:
            # Log other general exceptions
            logging.error("Unexpected error occurred: %s", e)

    def calculate_prototype_metrics(self):
        emb = self.encode_ref(self.model)
        p = PrototypeAnalyzer(emb, self.model.prototypes, self.ref.adata)
        return p.calculate_summary()

    def calculate_other_metrics(self):
        ref_emb = self.encode_adata(self.original_ref.adata, self.model)
        query_emb = self.encode_query(self.model)
        return {"propagation loss": self.model.propagation(ref_emb).cpu().item()}, {
            "propagation loss": self.model.propagation(query_emb).cpu().item()
        }

    def get_p(self, s):
        return F.softmax(s / self.temperature)

    def calculate_pair_matching(self, scores):
        def get_hard_cluster(tensor):
            # Find the indices of the maximum values along each row
            max_indices = torch.argmax(tensor, dim=1)

            # Create a one-hot tensor with the same shape as the input
            one_hot = torch.zeros_like(tensor)

            # Scatter 1s into the one-hot tensor at the max indices
            one_hot.scatter_(1, max_indices.unsqueeze(1), 1.0)
            return one_hot

        def calculate_entropy(tensor):
            """
            Calculate the entropy of a tensor.

            Parameters:
                tensor (torch.Tensor): Input tensor.

            Returns:
                float: Entropy of the tensor.
            """
            # Flatten the tensor and convert to probabilities
            flattened = tensor.flatten()
            probabilities = flattened / flattened.sum()

            # Ensure no zero values (to avoid log(0))
            probabilities = probabilities[probabilities > 0]

            # Compute entropy
            entropy = -torch.sum(probabilities * torch.log(probabilities))
            return entropy.item()

        score_t, score_s = (
            scores[: self.batch_size],
            scores[self.batch_size : 2 * self.batch_size],
        )
        p_t, p_s = self.get_p(score_t), self.get_p(score_s)
        h_t, h_s = get_hard_cluster(score_t), get_hard_cluster(score_s)
        c_t, c_s = torch.argmax(h_t, dim=1), torch.argmax(h_s, dim=1)
        matches = c_t == c_s
        num_matches = matches.sum().item()
        entropy = calculate_entropy(p_t.sum(0)) + calculate_entropy(p_s.sum(0))
        p_avg_entropy = -torch.sum(
            p_t * torch.log(p_t + 1e-9), dim=1
        )  # Add small epsilon to avoid log(0)

        return num_matches, entropy / 2, p_avg_entropy

    def freeze_prototypes(self, de_freeze=False):
        for name, param in self.model.named_parameters():
            if "prototypes" in name:
                param.requires_grad = de_freeze

    def train_one_epoch(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        cvae_losses = AverageMeter()
        swav_losses = AverageMeter()
        # prot_decoding_losses = AverageMeter()
        propagation_losses = AverageMeter()
        prot_emb_sim_losses = AverageMeter()
        num_match_avg, prob_entropy_avg, p_entropy_avg = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )

        self.model.train()
        use_the_queue = False

        end = time.time()
        for iteration, inputs in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            bs = inputs["x"].size(0)
            self.update_learning_rate(epoch, iteration)

            # move same functionality inside model

            with torch.no_grad():
                self.model.normalize_prototypes()

            inputs = self.move_input_on_device(inputs)
            inputs = reshape_and_reorder_dict(inputs)
            _, projector_out, scores, cvae_loss, prot_decoding_loss = self.model(inputs)
            propagation, prot_emb_sim = prot_decoding_loss

            projector_out = projector_out.detach()
            swav_loss = self.compute_swav_loss(projector_out, scores, bs, use_the_queue)
            num_match, prob_entropy, p_entropy = self.calculate_pair_matching(scores)
            loss = (
                swav_loss
                + cvae_loss * self.cvae_loss_scaler
                # + prot_decoding_loss * self.prot_decoding_loss_scaler
                + propagation * self.propagation_reg
                + prot_emb_sim * self.prot_emb_sim_reg
            )
            self.optimizer.zero_grad()

            if self.use_fp16:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.freeze_prototypes_nepochs > 0:
                if epoch < self.freeze_prototypes_nepochs:
                    self.freeze_prototypes()
                else:
                    self.freeze_prototypes(de_freeze=True)
            # if (
            #     epoch * len(self.train_loader) + iteration
            #     < self.freeze_prototypes_niters
            # ):
            #     for name, p in self.model.named_parameters():
            #         if "prototypes" in name:
            #             p.grad = None

            self.optimizer.step()

            losses.update(loss.item(), inputs["x"].size(0))
            cvae_losses.update(cvae_loss.item(), inputs["x"].size(0))
            swav_losses.update(swav_loss.item(), inputs["x"].size(0))
            # prot_decoding_losses.update(prot_decoding_loss.item(), inputs["x"].size(0))
            propagation_losses.update(propagation.item(), inputs["x"].size(0))
            prot_emb_sim_losses.update(prot_emb_sim.item(), inputs["x"].size(0))
            num_match_avg.update(num_match, bs)
            prob_entropy_avg.update(prob_entropy, bs)
            p_entropy_avg.update(p_entropy, bs)

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
            propagation_losses.avg,
            prot_emb_sim_losses.avg,
            num_match_avg.avg,
            prob_entropy_avg.avg,
            p_entropy_avg.avg,
        ), self.queue

    def update_learning_rate(self, epoch, iteration):
        iteration = epoch * len(self.train_loader) + iteration
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[iteration]

    def hard_clusters(self, out):
        def one_hot_max_tensor(tensor):
            """
            Convert each row of a tensor into a one-hot encoded row based on the maximum value in each row.

            Parameters:
            - tensor (torch.Tensor): Input 2D tensor of shape (b, p).

            Returns:
            - one_hot_tensor (torch.Tensor): One-hot encoded tensor of the same shape as the input.
            """
            # Find the indices of the maximum values along each row
            max_indices = torch.argmax(tensor, dim=1)

            # Create a zero tensor of the same shape as the input
            one_hot_tensor = torch.zeros_like(tensor)

            # Set the maximum indices to 1 in each row
            one_hot_tensor[torch.arange(tensor.size(0)), max_indices] = 1

            return one_hot_tensor

        return one_hot_max_tensor(out)

    def compute_swav_loss(self, embedding, scores, bs, use_the_queue):
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = scores[bs * crop_id : bs * (crop_id + 1)].detach()

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
                if self.hard_clustering == 1:
                    q = self.hard_clusters(q)

            subloss = 0
            # for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
            #     x = output[bs * v : bs * (v + 1)] / self.temperature
            #     subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = (
                    scores[bs * v : bs * (v + 1)] / self.temperature
                )  # logits for the v-th crop

                if self.loss_type == "kl1":
                    # KL divergence from q to p (KL(q || p))
                    p = F.log_softmax(x, dim=1)  # convert logits to probabilities
                    subloss += torch.mean(
                        torch.sum(
                            q * (torch.log(q + 1e-9) - torch.log(p + 1e-9)), dim=1
                        )
                    )
                elif self.loss_type == "kl2":
                    # KL divergence from p to q (KL(p || q))
                    p = F.log_softmax(x, dim=1)  # convert logits to probabilities
                    subloss += torch.mean(
                        torch.sum(
                            p * (torch.log(p + 1e-9) - torch.log(q + 1e-9)), dim=1
                        )
                    )
                else:
                    # Default cross-entropy functionality (unchanged)
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

    def get_model_prototypes(self, model):
        prototypes = model.get_prototypes()
        if self.use_projector_out:
            return model.projection_head(prototypes)
        else:
            return prototypes

    def plot_by_augmentation(self, n_samples, n_augmentations):
        print("using new plot augmentations, correct decoding")
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

        # Initialize lists to store embeddings, labels, and study labels
        all_embeddings = []
        all_labels = []
        all_celltypes = []
        all_study_labels = []

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

            # Append embeddings, labels, cell types, and study labels to the lists
            all_embeddings.append(embeddings)
            all_labels.append(labels)
            all_celltypes.append(inputs["celltypes"].cpu().numpy())
            all_study_labels.append(
                inputs["batch"].cpu().numpy()
            )  # Extract study labels

        # Concatenate all embeddings, labels, cell types, and study labels
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_celltypes = np.concatenate(all_celltypes, axis=0)
        all_study_labels = np.concatenate(
            all_study_labels, axis=0
        )  # Concatenate study labels

        # Create a reverse dictionary for cell type decoding
        all_celltypes = all_celltypes.reshape(-1)
        cell_type_encoder = self.train_ds.cell_type_encoder
        reverse_cell_type_encoder = {v: k for k, v in cell_type_encoder.items()}
        decoded_celltypes = np.array(
            [reverse_cell_type_encoder[idx] for idx in all_celltypes]
        )

        cell_umap, prototype_umap = calculate_umap(all_embeddings)
        plot_umap(
            cell_umap,
            prototype_umap,
            decoded_celltypes,
            all_study_labels.reshape(-1),  # Pass study labels to plot_umap
            all_labels.reshape(-1),  # Optional augmentation labels
        )

    def plot_projected_umap(self, save=True):
        self.use_projector_out = True
        ref = self.plot_ref_umap(save)
        query = self.plot_query_umap(save)
        self.use_projector_out = False
        return ref, query

    def additional_plots(self):
        if self.use_projector:
            return self.plot_projected_umap()

    def freeze_except_decoder(self, model):
        """
        Freeze all the weights of the model except those in the decoder.

        Args:
            model: The model whose weights need to be frozen.
        """
        # Iterate over all modules in the model
        for name, param in model.named_parameters():
            if "decoder" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def only_decoder_train(self):
        # Freeze all parts of the model except the decoder
        self.freeze_except_decoder(self.model)

        # Initialize a separate optimizer for the decoder parameters
        decoder_optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.base_lr,
            momentum=0.9,
            weight_decay=self.wd,
        )
        decoder_optimizer = LARC(
            optimizer=decoder_optimizer, trust_coefficient=0.001, clip=False
        )

        cvae_losses = AverageMeter()

        for epoch in range(self.pretraining_epochs):
            for iteration, inputs in enumerate(self.train_loader):
                inputs = self.move_input_on_device(inputs)
                inputs = reshape_and_reorder_dict(inputs)
                _, _, _, cvae_loss, _ = self.model(inputs)  # Modify as needed
                decoder_optimizer.zero_grad()

                if self.use_fp16:
                    with apex.amp.scale_loss(
                        cvae_loss, decoder_optimizer
                    ) as scaled_loss:
                        scaled_loss.backward()
                else:
                    cvae_loss.backward()

                decoder_optimizer.step()
                cvae_losses.update(cvae_loss.item(), inputs["x"].size(0))

            # Log the average loss for this epoch
            wandb.log({"decoder loss": cvae_losses.avg})

            logger.info(
                f"Epoch: [{epoch+1}/{self.pretraining_epochs}]\t"
                f"Decoder Loss {cvae_losses.val:.4f} ({cvae_losses.avg:.4f})"
            )

    def finetune(self):
        print(f"-------finetuning: {self.fine_tuning_epochs}----------")
        # old_aug_type = self.augmentation_type

        # scpoli_query = scPoli.load_query_data(
        #     adata=self.ref.adata,
        #     reference_model=self.get_scpoli(),
        #     labeled_indices=[],
        # )
        # self.model.set_scpoli_model(scpoli_query.model)
        self.model = self.adapt_ref_model(self.model, self.ref.adata)
        self.train_augmentation = "cell_type"
        self.build_data()
        self.build_optimizer()
        # self.setup()
        self.train(self.fine_tuning_epochs)
        # self.augmentation_type = old_aug_type

    def tune_nmb_crops(self, adata_list):
        max_nmb_crops = sys.maxsize

        for adata in adata_list:
            min_cell_cnt = adata.obs.cell_type.value_counts().min()
            max_nmb_crops = min(min_cell_cnt, max_nmb_crops)
        self.nmb_crops[0] = min(self.nmb_crops[0], max_nmb_crops)


if __name__ == "__main__":
    swav = SwAV()
    swav.setup()
    swav.run()
    swav.encode_ref()


# Example usage
# To run with command line arguments:
# python script.py --dataset some_dataset --dump_name_version 4 --nmb_crops 10 12 --augmentation_type knn --epochs 500
