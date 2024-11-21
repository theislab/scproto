from interpretable_ssl.trainers.trainer import Trainer
from interpretable_ssl.models.scpoli import *
from interpretable_ssl.loss_manager import *
from scarches.models.scpoli import scPoli
import numpy as np
from interpretable_ssl.train_utils import *
from interpretable_ssl.configs.defaults import *
from scarches.dataset.scpoli.anndata import MultiConditionAnnotatedDataset
import os
from interpretable_ssl.evaluation.visualization import *
from tqdm import tqdm
from interpretable_ssl.evaluation.metrics import MetricCalculator
from torch.utils.data import DataLoader
from interpretable_ssl.utils import log_time


class ScpoliTrainer(Trainer):
    # @log_time('scpoli trainer')
    def __init__(
        self, debug=False, dataset=None, ref_query=None, parser=None, **kwargs
    ) -> None:
        
        self.default_values = get_defaults().copy()
        self.update_kwargs(parser, kwargs)
        super().__init__(debug, dataset, ref_query, **kwargs)

    def update_kwargs(self, parser, kwargs):
        if parser is not None:
            parser = self.add_parser_args(parser)
            args = parser.parse_args()
            args_dict = vars(args)

            # Remove keys from args_dict if their value is the string "None"
            keys_to_remove = [
                key for key, value in args_dict.items() if value == "None"
            ]
            for key in keys_to_remove:
                del args_dict[key]

            kwargs.update(args_dict)

        # Use default values for any missing kwargs
        for key, value in self.default_values.items():
            # if value == "":
            #     value = None
            kwargs.setdefault(key, value)
        return kwargs

    def add_parser_args(self, parser):
        # Add arguments to parser with default values from dictionary
        for key, value in self.default_values.items():
            if isinstance(value, bool):
                # Handle boolean arguments with action='store_true'
                parser.add_argument(
                    f"--{key}",
                    action="store_true",
                    help=f"Set {key} to true (default is {value})",
                )
            else:
                # Handle other types of arguments
                arg_type = type(value) if value is not None else str
                if value == "":
                    value = None
                parser.add_argument(
                    f"--{key}",
                    type=arg_type,
                    default=value,
                    help=f"Set {key} (default is {value})",
                )
        return parser

    def load_model(self):

        model = self.get_model(self.ref.adata)
        path = self.get_model_path()
        model.load_state_dict(torch.load(path)["model_state_dict"])
        return model

    def load_query_model(self, adata=None):
        if adata is None:
            adata = self.query.adata
        model = self.load_model()
        model = self.adapt_ref_model(model, adata)
        model.to(self.device)
        return model

    def adapt_ref_model(self, ref_model, adata):
        scpoli_query = scPoli.load_query_data(
            adata=adata,
            reference_model=self.get_scpoli(ref_model, False),
            labeled_indices=[],
        )
        ref_model.set_scpoli_encoder(scpoli_query.model)
        ref_model.to(self.device)
        return ref_model
        
    def move_input_on_device(self, inputs):
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def prepare_scpoli_dataloader(self, adata, scpoli_model, shuffle=True):

        dataset = MultiConditionAnnotatedDataset(
            adata,
            condition_keys=[self.condition_key],
            # cell_type_keys=[self.cell_type_key],
            condition_encoders=scpoli_model.condition_encoders,
            conditions_combined_encoder=scpoli_model.conditions_combined_encoder,
            # cell_type_encoder=scpoli_model.cell_type_encoder,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=scpoli_utils.custom_collate,
            shuffle=shuffle,
        )
        return loader

    def encode_batch(self, model, batch):
        pass

    def get_scpoli(self, pretrained_model, return_model=True):
        pass

    def encode_ref(self, model=None):
        return self.encode_adata(self.ref.adata, model)

    def encode_query(self, ref_model = None):
        if ref_model is None:
            model = self.load_query_model()
        else:
            model = self.adapt_ref_model(ref_model, self.query.adata)
        return self.encode_adata(self.query.adata, model)

    def encode_adata(self, adata, model=None):
        if model is None:
            model = self.load_model()
        loader = self.prepare_scpoli_dataloader(
            adata, self.get_scpoli(model), shuffle=False
        )
        embeddings = [self.encode_batch(model, batch) for batch in tqdm(loader)]
        return torch.cat(embeddings)

    def get_model_prototypes(self, model):
        return None

    def calculate_umaps(self, trained_model=True):
        if trained_model:
            model = self.load_model()
        else:
            model = self.get_model()
            model.to("cuda")
        embeddings = self.encode_ref(model)
        prototypes = self.get_model_prototypes(model)

        self.embedding_umap, self.prototype_umap = calculate_umap(
            embeddings, prototypes
        )

    def get_umap_path(self, data_part="ref"):
        pass

    def plot_umap(self, model, adata, split, save_plot=True):
        latent = self.encode_adata(adata, model)
        prototypes = self.get_model_prototypes(model)
        latent_umap, prototype_umap = calculate_umap(latent, prototypes)
        obs = adata.obs
        return plot_3umaps(
            latent_umap,
            prototype_umap,
            obs.cell_type,
            obs.study,
            save_plot,
            self.get_umap_path(split),
        )

    def plot_ref_umap(self, save_plot=True, name_postfix=None, model=None):
        
        if model is None:
            model = self.load_model()
        if name_postfix is not None:
            name = f"ref-{name_postfix}"
        else:
            name = f'ref'
        return self.plot_umap(model, self.ref.adata, name, save_plot)

    def plot_query_umap(self, save_plot=True):
        model = self.load_query_model()
        return self.plot_umap(model, self.query.adata, "query", save_plot)

    def additional_plots(self):
        pass

    def calculate_other_metrics(self):
        return {}, {}
    
    def save_metrics(self):
        ref_other, query_other = self.calculate_other_metrics()
        ref_latent = self.encode_ref(self.model)
        MetricCalculator(
            self.ref.adata, [ref_latent], self.dump_path,
            save_path=self.get_metric_file_path("ref")
        ).calculate(ref_other)
        # calculate_scib_metrics_using_benchmarker(
        #     self.ref.adata, ref_latent, self.get_scib_file_path('ref')
        # )
        query_latent = self.encode_query(self.model)
        MetricCalculator(
            self.query.adata,
            [query_latent],
            self.dump_path,
            save_path=self.get_metric_file_path("query"),
        ).calculate(query_other)
        # calculate_scib_metrics_using_benchmarker(
        #     self.query.adata, query_latent, self.get_scib_file_path('query')
        # )
