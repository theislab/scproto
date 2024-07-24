from interpretable_ssl.trainers.trainer import Trainer
from interpretable_ssl.models.scpoli import *
from sklearn.model_selection import train_test_split
from interpretable_ssl.datasets.immune import ImmuneDataset
from interpretable_ssl.train_utils import optimize_model
import sys
from interpretable_ssl.evaluation.visualization import plot_umap
from interpretable_ssl.loss_manager import *
from interpretable_ssl import utils
from torch.utils.data import WeightedRandomSampler
from scarches.models.scpoli import scPoli
import numpy as np
from interpretable_ssl.train_utils import *
from interpretable_ssl.configs.defaults import get_scpoli_defaults
from scarches.dataset.scpoli.anndata import MultiConditionAnnotatedDataset
from torch.utils.data import DataLoader
from interpretable_ssl.models.linear import LinearClassifier
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
import pandas as pd
import apex

class ScpoliTrainer(Trainer):
    def __init__(self, parser=None, **kwargs) -> None:
        self.default_values = get_scpoli_defaults()
        self.update_kwargs(parser, kwargs)

        # Extract ScpoliTrainer-specific arguments
        condition_key = kwargs.pop(
            "condition_key", self.default_values["condition_key"]
        )
        cell_type_key = kwargs.pop(
            "cell_type_key", self.default_values["cell_type_key"]
        )

        super().__init__(**kwargs)

        self.condition_key = condition_key
        self.cell_type_key = cell_type_key

    def update_kwargs(self, parser, kwargs):
        if parser is not None:
            parser = self.add_parser_args(parser)
            args = parser.parse_args()
            kwargs.update(vars(args))

        # Use default values for any missing kwargs
        for key, value in self.default_values.items():
            if value == "":
                value = None
            kwargs.setdefault(key, value)
        return kwargs

    def add_parser_args(self, parser):
        # Add arguments to parser with default values from dictionary
        for key, value in self.default_values.items():
            if isinstance(value, bool):
                # Handle boolean arguments with action='store_true'
                parser.add_argument(f"--{key}", action='store_true', help=f"Set {key} to true (default is {value})")
            else:
                # Handle other types of arguments
                arg_type = type(value) if value is not None else str
                if value == '':
                    value = None
                parser.add_argument(f"--{key}", type=arg_type, default=value, help=f"Set {key} (default is {value})")
        return parser

    def split_train_test(self, ref):
        train_idx, val_idx = train_test_split(range(len(ref.adata)))
        train, val = ref.adata[train_idx], ref.adata[val_idx]
        return train, val

    def to_save(self, val_loss, best_val_loss):
        if val_loss.overal < best_val_loss:
            best_val_loss = val_loss.overal
            return True
        return False

    def train_step(self, model, optimizer, train_adata):
        return scpoli_train_step(model, train_adata, optimizer, self.batch_size)

    def test_step(self, model, val_adata):
        return scpoli_test_step(model, val_adata, self.batch_size)

    def train(self, epochs):
        print("running scpoli trainer class train")
        ref, _ = self.ref, self.query
        model = self.get_model(ref.adata)
        model.to(self.device)

        train_adata, val_adata = self.split_train_test(ref)

        # init training parameter and wandb
        optimizer = self.get_optimizer(model)
        model_path = self.get_model_path()

        # init wandb
        self.init_wandb(model_path, len(train_adata), len(self.val_adata))
        best_val_loss = sys.maxsize

        for epoch in range(epochs):

            train_loss = self.train_step(model, optimizer, train_adata)
            val_loss = self.test_step(model, val_adata)
            self.log_loss(train_loss, val_loss)
            if self.to_save(val_loss, best_val_loss):
                utils.save_model_checkpoint(
                    model,
                    epoch,
                    model_path,
                )
        return train_loss.overal, best_val_loss

    def load_model(self):

        model = self.get_model(self.ref.adata)
        path = self.get_model_path()
        model.load_state_dict(torch.load(path)["model_state_dict"])
        return model

    def get_representation(self, model, adata):
        return model.get_representation(adata)

    def visualize(self):
        model = self.load_model()
        model.to(self.device)
        # get latent representation of reference data
        data_latent_source = self.get_representation(model, self.ref.adata)
        self.ref.adata.obsm[f"{self.experiment_name}"] = data_latent_source

        plot_umap(self.ref.adata, f"{self.experiment_name}")

    def get_pretraining_epochs(self, epochs):
        pretraining_epochs = epochs - self.fine_tuning_epochs
        if pretraining_epochs >= 0:
            return pretraining_epochs
        return 0

    def get_query_model(self):
        model = self.load_model()
        scpoli_query = scPoli.load_query_data(
            adata=self.query.adata,
            reference_model=model.scpoli,
            labeled_indices=[],
        )
        model.set_scpoli_model(scpoli_query)
        return model

    def finetune_query_model(self, model, epochs=100):
        model.scpoli.train(
            n_epochs=self.fine_tuning_epochs,
            pretraining_epochs=self.get_pretraining_epochs(epochs),
            eta=10,
        )
        utils.save_model(model, self.get_query_model_path())
        return model

    def get_query_model_latent(self, model, adata):
        return model.scpoli.get_latent(adata, mean=True)

    # load pretrained model
    # k fold
    # train linear classifier on top of frozen features
    # save classification results (train/test)

    # agg results
    # save results
    def move_input_on_device(self, inputs):
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def prepare_scpoli_dataloader(self, adata, scpoli_model):
        dataset = MultiConditionAnnotatedDataset(
            adata,
            condition_keys=[self.condition_key],
            cell_type_keys=[self.cell_type_key],
            condition_encoders=scpoli_model.condition_encoders,
            conditions_combined_encoder=scpoli_model.conditions_combined_encoder,
            cell_type_encoder=scpoli_model.cell_type_encoder,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=scpoli_utils.custom_collate,
            shuffle=True,
        )
        return loader

    def classification_metrics(self, loader, encoder, model, class_names):
        model.to(self.device)
        model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in loader:
                batch_X = self.encode_batch(encoder, batch)
                outputs = model(batch_X)
                _, batch_preds = torch.max(outputs, 1)
                preds.extend(batch_preds.cpu().numpy())
                targets.extend(batch["celltypes"].cpu().numpy())

        report = classification_report(
            targets, preds, output_dict=True
        )
        return {class_names.get(key, key): value for key, value in report.items()}

    def linear_evaluation_ssl(
        self, pretrained_model, train_adata, test_adata, epochs=300
    ):
        scpoli_model = self.get_scpoli_model(pretrained_model)
        # define data laoder
        train_loader = self.prepare_scpoli_dataloader(train_adata, scpoli_model)

        # define model, loss, opt
        model = LinearClassifier(self.latent_dims, self.dataset.num_classes)
        model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        # model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")

        for epoch in range(epochs):
            # for each batch
            for batch in train_loader:
                # encode data
                batch_X = self.encode_batch(pretrained_model, batch)

                # train step
                outputs = model(batch_X)
                loss = criterion(outputs, batch["celltypes"].squeeze())
                wandb.log({'epoch': epoch, 'loss': loss})
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        test_laoder = self.prepare_scpoli_dataloader(test_adata, scpoli_model)
        class_names = {str(value): key for key, value in scpoli_model.cell_type_encoder.items()}
        return self.classification_metrics(
            test_laoder,
            pretrained_model,
            model,
            class_names
        )

    def save_classification_results(self, results):

        # Initialize dictionaries to store DataFrames for each metric
        metrics_dfs = {
            metric: [] for metric in ["precision", "recall", "f1-score", "support"]
        }
        accuracy_list = []

        # Convert the results to DataFrames
        for fold_result in results:
            accuracy_list.append(
                {"fold": fold_result["fold"], "accuracy": fold_result["accuracy"]}
            )
            for metric in metrics_dfs.keys():
                metric_data = {
                    class_name: fold_result[class_name][metric]
                    for class_name in fold_result.keys()
                    if class_name not in ["accuracy", "fold"]
                }
                metric_data["fold"] = fold_result["fold"]
                metrics_dfs[metric].append(metric_data)

        # Create DataFrames for each metric and calculate averages
        for metric in metrics_dfs.keys():
            df = pd.DataFrame(metrics_dfs[metric])
            df.loc["average"] = df.mean(numeric_only=True)
            metrics_dfs[metric] = df
        # Create DataFrame for accuracy and calculate average
        accuracy_df = pd.DataFrame(accuracy_list)
        accuracy_df.loc['average'] = accuracy_df.mean(numeric_only=True)
        
        save_path = self.get_dump_path()
        # Save to Excel
        with pd.ExcelWriter(save_path + "/linear_classification.xlsx") as writer:
            
            for metric, df in metrics_dfs.items():
                df.to_excel(writer, sheet_name=metric, index=True)
            accuracy_df.to_excel(writer, sheet_name='accuracy', index=True)

    def kfold_linear_evaluation(self, k_folds=5):
        pretrained_model = self.load_model()
        adata = self.ref.adata
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(range(len(adata)))):
            train_adata, test_adata = adata[train_idx], adata[test_idx]
            report = self.linear_evaluation_ssl(
                pretrained_model, train_adata, test_adata
            )
            report["fold"] = fold + 1
            results.append(report)
            self.save_classification_results(results)

    def encode_batch(self, model, batch):
        pass

    def get_scpoli_model(self, pretrained_model):
        pass
    
    def setup(self):
        pass
    
    def run(self):
        if not self.only_eval:
            self.train(self.epochs)
        if self.linear_eval:
            self.kfold_linear_evaluation()


class ScpoliProtBarlowTrainer(ScpoliTrainer):
    def __init__(self, dataset, projection_version=0) -> None:
        super().__init__(dataset)
        self.experiment_name = "barlow"
        self.projection_version = projection_version

    def get_model_name(self):
        name = super().get_model_name()
        if self.projection_version != 0:
            name = f"{name}_projection-version-{self.projection_version}"
        return name

    def get_model(self, adata):
        return BarlowPrototypeScpoli(
            adata, self.latent_dims, self.num_prototypes, self.projection_version
        )


class LinearTrainer(ScpoliTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.num_prototypes = 32
        self.experiment_name = "linear-prot-scpoli-task_ratio10"

    def get_model(self, adata):
        head = nn.Linear(self.num_prototypes, self.dataset.num_classes, bias=False)
        return LinearPrototypeScpoli(adata, self.latent_dims, self.num_prototypes, head)

    def train_step(self, model, train_loader, optimizer):
        total_loss = PrototypeLoss()
        for scpoli_batch in train_loader:
            batch_loss = model(scpoli_batch)
            total_loss += batch_loss
            optimize_model(batch_loss, optimizer)
        total_loss.normalize(len(train_loader))
        return total_loss

    def test_step(self, model, test_loader):
        model.eval()
        test_loss = PrototypeLoss()
        with torch.inference_mode():
            for scpoli_batch in test_loader:
                test_loss += model(scpoli_batch)
        test_loss.normalize(len(test_loader))
        return test_loss


class BarlowTrainer(ScpoliTrainer):
    def __init__(self) -> None:
        print("running barlow-scpoli")
        super().__init__()
        self.experiment_name = "barlow-scpoli"

    def get_model(self, adata):
        return BarlowScpoli(adata, self.latent_dims)


class ScpoliOriginal(ScpoliTrainer):

    def __init__(self) -> None:
        super().__init__()
        condition_key = "study"
        cell_type_key = "cell_type"
        self.scpoli_trainer = scPoli(
            adata=self.ref.adata,
            condition_keys=condition_key,
            cell_type_keys=cell_type_key,
            latent_dim=self.latent_dims,
            recon_loss="nb",
        )

    def get_model(self, adata):

        return self.scpoli_trainer.model

    def train(self):
        self.scpoli_trainer.train(
            n_epochs=100,
            pretraining_epochs=50,
        )
        path = self.get_model_path()
        utils.save_model(self.scpoli_trainer.model, path)

    def get_representation(self, model, adata):
        return self.scpoli_trainer.get_latent(adata, mean=True)

    def get_model_name(self):
        return f"original-scpoli-latent{self.latent_dims}"


class SimClrTrainer(ScpoliTrainer):
    def __init__(self, dataset=None) -> None:
        super().__init__(dataset)
        self.experiment_name = "simclr"

    def get_model(self, adata):
        return SimClrPrototype(adata, self.latent_dims, self.num_prototypes)
