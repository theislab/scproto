from scib_metrics.benchmark import Benchmarker
import scanpy as sc
from copy import deepcopy
def calculate_scib_metrics(input_adata, latent, batch_key="batch", label_key="cell_type"):
    adata = deepcopy(input_adata)
    adata.obsm['latent'] = latent
    
    # Initialize the Benchmarker
    benchmarker = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=['latent'],
        n_jobs=1  # Adjust the number of jobs according to your system
    )

    # Perform the benchmark
    benchmarker.benchmark()

    # Get the results
    results = benchmarker.get_results(min_max_scale=False)
    return results, benchmarker