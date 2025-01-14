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
from interpretable_ssl.evaluation.cd4_marker import assign_prototype_labels

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
        model = self.get_model()
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
        query_model = self.get_model() # Create an uninitialized model of the same type
        query_model.load_state_dict(ref_model.state_dict())
        scpoli_query = scPoli.load_query_data(
            adata=adata,
            reference_model=self.get_scpoli(query_model, False),
            labeled_indices=[],
        )
        query_model.set_scpoli_encoder(scpoli_query.model)
        query_model.to(self.device)
        return query_model

    def move_input_on_device(self, inputs):
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    def encode_batch(self, model, batch, return_maped=False, return_mapped_idx=True):
        batch = self.move_input_on_device(batch)
        model.eval()
        with torch.no_grad():
            encoder_out, x, x_mapped = model.encode(batch)
        # if self.use_projector_out:
        #     return x
        # else:
        if return_maped:
            if return_mapped_idx:
                return torch.argmax(x_mapped, dim=1)
            else:
                return x_mapped
            
            # return x_mapped
        return encoder_out

    def get_scpoli(self, pretrained_model, return_model=True):
        if return_model:
            return pretrained_model.scpoli_encoder
        return pretrained_model.scpoli_

    def encode_ref(self, model=None):
        return self.encode_adata(self.ref.adata, model)

    def encode_query(self, ref_model=None):
        if ref_model is None:
            model = self.load_query_model()
        else:
            model = self.adapt_ref_model(ref_model, self.query.adata)
        return self.encode_adata(self.query.adata, model)

    def encode_adata(self, adata, model=None, return_mapped=False, return_mapped_idx=True):
        if model is None:
            model = self.load_model()
        loader = self.prepare_scpoli_dataloader(
            adata, self.get_scpoli(model), shuffle=False
        )
        embeddings = [self.encode_batch(model, batch, return_mapped, return_mapped_idx) for batch in tqdm(loader)]
        return torch.cat(embeddings)

    def get_model_prototypes(self, model):
        return None

    def get_umap_path(self, data_part="ref"):
        pass

    def plot_umap(self, model, adata, split, save_plot=True):
        latent = self.encode_adata(adata, model)
        prototypes = self.get_model_prototypes(model)
        latent_umap, prototype_umap = calculate_umap(latent, prototypes)
        obs = adata.obs
        if prototypes is not None:
            prototype_assignments = self.encode_adata(adata, model, True)
            proto_df = assign_prototype_labels(adata, prototype_assignments, self.num_prototypes)
            proto_labels = proto_df.prototype_label
        else:
            proto_labels = None
        return plot_3umaps(
            latent_umap,
            prototype_umap,
            obs.cell_type,
            obs.study,
            proto_labels,
            save_plot,
            self.get_umap_path(split),
        )

    def plot_ref_umap(self, save_plot=True, name_postfix=None, model=None):

        if model is None:
            model = self.load_model()
        if name_postfix is not None:
            name = f"ref-{name_postfix}"
        else:
            name = f"ref"
        return self.plot_umap(model, self.ref.adata, name, save_plot)

    def plot_query_umap(self, save_plot=True):
        model = self.load_query_model()
        return self.plot_umap(model, self.query.adata, "query", save_plot)

    def additional_plots(self):
        pass

    def calculate_other_metrics(self):
        return {}, {}

    def save_metrics(self, save=True, calc_others = True):
        if calc_others:
            ref_other, query_other = self.calculate_other_metrics()
        else:
            ref_other, query_other = {}, {}
        ref_latent = self.encode_ref(self.model)
        ref_df = MetricCalculator(
            self.ref.adata,
            [ref_latent],
            self.dump_path,
            save_path=self.get_metric_file_path("ref"),
        ).calculate(ref_other, save)
        # calculate_scib_metrics_using_benchmarker(
        #     self.ref.adata, ref_latent, self.get_scib_file_path('ref')
        # )
        query_latent = self.encode_query(self.model)
        query_df = MetricCalculator(
            self.query.adata,
            [query_latent],
            self.dump_path,
            save_path=self.get_metric_file_path("query"),
        ).calculate(query_other, save)
        def get_metrics(df):
            return df.loc['latent' ,['Batch correction','Bio conservation', 'scib total']], df[['Rank-PCA', 'Corr-PCA', 'Corr-Weighted']].mean(axis=1).iloc[0].item()
        return get_metrics(ref_df), get_metrics(query_df)
    
        # calculate_scib_metrics_using_benchmarker(
        #     self.query.adata, query_latent, self.get_scib_file_path('query')
        # )
