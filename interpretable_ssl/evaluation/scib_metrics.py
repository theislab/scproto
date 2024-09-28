import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from scib_metrics.benchmark import Benchmarker

def calculate_scib_metrics_using_benchmarker(input_adata, latent, save_path=None, batch_key="batch", label_key="cell_type"):
    # Convert the latent tensor to a numpy array if it's a PyTorch tensor
    if isinstance(latent, torch.Tensor):
        latent = latent.detach().cpu().numpy()

    # Create a deep copy of the input AnnData object
    adata = deepcopy(input_adata)
    
    # Store the latent embeddings in the AnnData object
    adata.obsm['latent'] = latent
    
    # Initialize the Benchmarker
    benchmarker = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=['latent'],
        n_jobs=-1  # Adjust the number of jobs according to your system
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
