from experiment_runner import (
    ExperimentRunner,
)  # Assuming ExperimentRunner is in a file named experiment_runner.py
from interpretable_ssl.trainers.swav import SwAV  # Import your SwAV model class
from interpretable_ssl.trainers.scpoli_original import (
    OriginalTrainer,
)  # Import your ScPoli model class (if needed)
import itertools
import os
import pandas as pd
from interpretable_ssl.evaluation.metrics import *
from tqdm import tqdm
import psutil
from interpretable_ssl.trainers.base import TrainerBase

from interpretable_ssl.evaluation.metric_generator import *
from interpretable_ssl.datasets.immune import *


class ExperimentEvaluator(ExperimentRunner):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        if self.dataset is not None:
            self.ref_query = self.dataset.get_train_test()
        else:
            self.ref_query = None

        self.trainers = []

    def create_trainer(self, params, model_type):
        params["debug"] = True
        params["dataset"] = self.dataset
        params["ref_query"] = self.ref_query
        """Create a trainer instance based on the model type and parameters."""
        if params.get("experiment_name") is None:
            params["experiment_name"] = model_type
        if model_type == "swav":
            trainer = SwAV(**params)
        elif model_type == "scpoli":
            trainer = OriginalTrainer(**params)
            #     latent_dims=params.get(
            #         "latent_dims", self.original_defaults["latent_dims"]
            #     ),
            #     batch_size=params.get(
            #         "batch_size", self.original_defaults["batch_size"]
            #     ),
            #     debug=params.get("debug", True),  # Assuming debug=True is a default
            #     experiment_name=params.get(
            #         "experiment_name", self.original_defaults["experiment_name"]
            #     ),
            #     dataset_id=params.get(
            #         "dataset_id", self.original_defaults["dataset_id"]
            #     ),
            #     dataset=self.dataset,
            #     ref_query=self.ref_query,
            #     # no_data='True',
            #     model_name_version=params.get(
            #         "model_name_version", self.original_defaults["model_name_version"]
            #     ),
            # )
        else:
            raise ValueError("Unsupported model type: {}".format(model_type))

        trainer.name = self.generate_job_name(params, model_type)
        return trainer

    def generate_job_name(self, params, model_type):
        job_name = super().generate_job_name(params)
        return job_name

    def generate_trainers(self, item_list):
        """Generate a list of trainer instances based on the parameter grid."""
        all_trainers = []

        for item_to_test in item_list:
            trainer_list = []
            for model_type, model_params in item_to_test.items():
                keys, values = zip(*model_params.items())

                # Set up tqdm progress bar
                total_combinations = len(list(itertools.product(*values)))
                with tqdm(
                    total=total_combinations,
                    desc=f"Generating trainers for {model_type}",
                ) as pbar:
                    for value_combination in itertools.product(*values):
                        params = dict(zip(keys, value_combination))

                        # Generate a trainer based on the current parameters
                        trainer = self.create_trainer(params, model_type)
                        trainer_list.append(trainer)

                        # Get CPU and memory usage
                        memory_info = psutil.virtual_memory()
                        memory_used_percent = memory_info.percent
                        cpu_used_percent = psutil.cpu_percent()

                        # Update tqdm progress bar with CPU and memory usage
                        pbar.set_postfix(
                            {
                                "Memory": f"{memory_used_percent}%",
                                "CPU": f"{cpu_used_percent}%",
                                "trainer_name": trainer.name,
                            }
                        )

                        # Update progress bar
                        pbar.update(1)

            all_trainers += trainer_list

        return all_trainers

    def evaluate_results(self, item_list, semi_supervised=True):
        
        trainers = self.generate_trainers(item_list)
        for t in trainers:
            print(t.name)
        # for trainer in trainers:
        #     if trainer.name[-4:] == "scpo":
        #         trainer.name = trainer.name[-4:] + trainer.experiment_name

        def filter_trainers(inp_trainers):
            def model_exist(trainer):
                if os.path.exists(trainer.get_model_path()):
                    return True
                return False

            return [trainer for trainer in inp_trainers if model_exist(trainer)]

        def get_results(finetuned=True):
            if semi_supervised:
                for trainer in trainers:
                    if "scpoli" in trainer.name:
                        continue
                    if trainer.fine_tuning_epochs != 0:
                        trainer.finetuning = finetuned
                    if not finetuned:
                        trainer.name += "_pretrained"

            if not finetuned:
                selected_trainers = [t for t in trainers if "scpoli" not in t.name]
            else:
                selected_trainers = trainers

            valid_trainers = filter_trainers(selected_trainers)
            self.trainers = valid_trainers
            print(
                f"all trainers: {len(trainers)}, valid trainers: {len(valid_trainers)}"
            )
            # invalid_trainers = set(trainers) - set(valid_trainers)
            # for trainer in invalid_trainers:
            #     print
            dfs = [MetricGenerator(t).generate_metrics() for t in valid_trainers]
            print("dfs size:", len(dfs))
            return dfs

        finetuned_dfs = get_results()
        pretrained_dfs = get_results(False)
        dfs = finetuned_dfs + pretrained_dfs

        res = pd.concat(dfs, axis=0)[
            [
                "scib total",
                "Batch correction",
                "Bio conservation",
                "knn f1 macro",
                "knn f1 micro",
                "knn f1 weighted",
                "linear classifier f1 macro",
                "linear classifier f1 micro",
                "linear classifier f1 weighted",
                "Rank-PCA",
                "Corr-PCA",
                "Corr-Weighted",
            ]
        ]
        df = res.round(5).drop_duplicates()
        # df.index = [idx.replace('_aug_comm', '') if 'scpoli' in idx else idx for idx in df.index]

        scpoli_df = df[df.index.str.contains("scpoli")]
        threshold = scpoli_df["scib total"].max()

        # df = df[df["scib total"] >= threshold]

        def highlight_greater(s):
            maxidx = scpoli_df["scib total"].idxmax()
            row_to_compare = df.loc[maxidx]
            return [
                "font-weight: bold" if val > row_to_compare[i] else ""
                for i, val in enumerate(s)
            ]

        df_styled = df.sort_values(["scib total"], ascending=False).style.apply(
            highlight_greater, axis=1
        )
        return df_styled.format("{:.3f}")


if __name__ == "__main__":
    ds = ImmuneDataset()
    evaluator = ExperimentEvaluator(ds)

    item_to_test = {
        "swav": {
            "dimensionality_reduction": ["pca", None],
            "num_prototypes": [100, 300, 500],
            "latent_dims": [8, 16],
            "augmentation_type": ["knn", "scanpy_knn", "community"],
        },
        "scpoli": {
            "latent_dims": [8, 16, 32],
            "batch_size": [512, 1024],
            "debug": [True, False],
        },
    }

    # Generate trainers based on parameter combinations
    # trainers = evaluator.generate_trainers([item_to_test])

    evaluator.evaluate_results([item_to_test])
