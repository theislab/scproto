import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from scib_metrics.benchmark import Benchmarker
from interpretable_ssl.models.linear import *
from interpretable_ssl.evaluation.knn import *

import sys

sys.path.append("/home/icb/fatemehs.hashemig/Islander/src")
from scGraph import *
import os

class MetricCalculator:
    def __init__(self, input_adata, latents, dump_path, keys=["latent"], save_path=None) -> None:

        self.batch_key = "batch"
        self.label_key = "cell_type"
        self.save_path = save_path
        self.dump_path = dump_path
        self.keys = keys
        self.adata = self.prepare_adata(input_adata, latents)
        # self.latents = latents

    def prepare_adata(self, input_adata, latents):

        # Create a deep copy of the input AnnData object
        adata = input_adata
        for key, latent in zip(self.keys, latents):
            # print(key)
            # Convert the latent tensor to a numpy array if it's a PyTorch tensor
            if isinstance(latent, torch.Tensor):
                latent = latent.detach().cpu().numpy()
                # Store the latent embeddings in the AnnData object
            adata.obsm[key] = latent
        adata = self.remove_single_sample_celltypes(adata)
        return adata

    def remove_single_sample_celltypes(self, adata):
        """
        Removes cells from AnnData where the cell type has only one sample.

        Parameters:
        - adata: AnnData object
        - celltype_column: The column name in adata.obs that contains cell type information (default: 'cell_type')

        Returns:
        - adata_filtered: AnnData object with filtered cells
        """
        # Count the occurrences of each cell type
        celltype_counts = adata.obs[self.label_key].value_counts()

        # Identify cell types that have more than one sample
        valid_celltypes = celltype_counts[celltype_counts > 1].index

        # Filter adata to keep only cells with valid cell types
        adata_filtered = adata[adata.obs[self.label_key].isin(valid_celltypes)].copy()
        
        return adata_filtered

    def calculate_scib(self):
        # Initialize the Benchmarker
        benchmarker = Benchmarker(
            self.adata,
            batch_key=self.batch_key,
            label_key=self.label_key,
            embedding_obsm_keys=self.keys,
        )

        # Perform the benchmark
        benchmarker.benchmark()
        # Get the results as a dictionary
        results_df = benchmarker.get_results(min_max_scale=False)
        return results_df

    def check_duplicate_category_adata(self, adata):
        for col in adata.obs.columns:
            if adata.obs[col].dtype.name == "category":
                print(col, adata.obs[col].cat.categories.duplicated().sum())

    def calculate_scgraph(self):
        
        adata_tmp_path = os.path.join(self.dump_path, "adata_tmp.h5ad")
        self.check_duplicate_category_adata(self.adata)
        self.adata.write(adata_tmp_path)

        scgraph = scGraph(
            adata_path=adata_tmp_path,
            batch_key="batch",
            label_key="cell_type",
            hvg=False,
            trim_rate=0.05,
            thres_batch=100,
            thres_celltype=10,
        )
        return scgraph.main(_obsm_list=self.keys)

    def extract_f1_score(self, df, key, prefix=""):
        return {
            f"{prefix} macro": df.loc[df["Class"] == "micro", "F1 Score"].values[0],
            f"{prefix} micro": df.loc[df["Class"] == "macro", "F1 Score"].values[0],
            f"{prefix} weighted": df.loc[df["Class"] == "weighted", "F1 Score"].values[
                0
            ],
            "model": key,
        }

    def linear_results(self, epochs=100):
        def get_linear_results(embedding, key):
            classifier = LinearClassifier(
                embedding, self.adata.obs[self.label_key], batch_size=128, epochs=epochs
            )
            classifier.train()
            df, _ = classifier.evaluate()

            return self.extract_f1_score(df, key, 'linear classifier f1')

        return self.get_results(get_linear_results)

    def knn_results(self):
        def get_knn_results(emb, key):
            scores = knn_classifier_with_f1_report(emb, self.adata.obs[self.label_key])
            return self.extract_f1_score(scores, key, "knn f1")

        return self.get_results(get_knn_results)

    def get_results(self, get_res_func):
        # score_list = [
        #     get_res_func(emb, key) for emb, key in zip(self.adata.obsm, self.keys)
        # ]

        score_list = []
        
        for key in self.keys:
            emb = self.adata.obsm[key]
            score_list.append(get_res_func(emb, key))
            
        # Create a DataFrame from the list of F1 scores
        result_df = pd.DataFrame(score_list)

        # Set the model name as the index
        result_df.set_index("model", inplace=True)

        return result_df

    def save(self, results):
        # Save the results to a CSV file if save_path is provided
        if self.save_path is not None:
            results.to_csv(self.save_path, index=False)
            print(f"Results saved to {self.save_path}")

    def concat_results(self, scib_res, scg_res):
        scib_clean = scib_res.loc[scib_res.index != "Metric Type"]
        scib_clean = scib_clean.rename(columns={"Total": "scib total"})
        result = pd.concat([scib_clean, scg_res], axis=1, join="inner")
        return result

    def calculate(self, other_metrics={}, save=True):
        scib_res = self.calculate_scib()
        scgraph_res = self.calculate_scgraph()
        final_res = self.concat_results(scib_res, scgraph_res)
        for key, val in other_metrics.items():
            final_res[key] = val
        if save:
            self.save(final_res)
        return final_res


def calculate_scib_metrics_using_benchmarker(
    input_adata, latent, save_path=None, batch_key="batch", label_key="cell_type"
):
    # Convert the latent tensor to a numpy array if it's a PyTorch tensor
    if isinstance(latent, torch.Tensor):
        latent = latent.detach().cpu().numpy()

    # Create a deep copy of the input AnnData object
    adata = deepcopy(input_adata)

    # Store the latent embeddings in the AnnData object
    adata.obsm["latent"] = latent

    # Initialize the Benchmarker
    benchmarker = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=["latent"],
        n_jobs=-1,  # Adjust the number of jobs according to your system
    )

    # Perform the benchmark
    benchmarker.benchmark()

    # Get the results as a dictionary
    results_df = benchmarker.get_results(min_max_scale=False)

    # Save the results to a CSV file if save_path is provided
    if save_path is not None:
        results_df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

    return results_df
