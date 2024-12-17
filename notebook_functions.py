# input adata, plot umap color by cell type

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from matplotlib.patches import Patch
import scanpy as sc
import torch
import torch.nn.functional as F
from scarches.models.scpoli._utils import one_hot_encoder
from torch.distributions import NegativeBinomial
import umap
from interpretable_ssl.evaluation.cd4_marker import *


def calculate_joint_umap(
    cells_tensor, proto_tensor, n_neighbors=15, min_dist=0.5, n_components=2
):
    """
    Calculates a joint UMAP for cells and prototypes.

    Parameters:
        cells_tensor: Tensor (NumPy array or PyTorch tensor) for cells.
        proto_tensor: Tensor (NumPy array or PyTorch tensor) for prototypes.
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance for UMAP embedding.
        n_components: Number of dimensions for the UMAP embedding.

    Returns:
        cell_umap: UMAP embedding for cells.
        proto_umap: UMAP embedding for prototypes.
    """
    import torch

    # Ensure tensors are on CPU and convert to NumPy
    if torch.is_tensor(cells_tensor):
        cells_tensor = cells_tensor.cpu().numpy()
    if torch.is_tensor(proto_tensor):
        proto_tensor = proto_tensor.cpu().numpy()

    # Combine cells and prototypes into a single array
    combined_data = np.vstack([cells_tensor, proto_tensor])

    # Run UMAP
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
        metric="cosine",
    )
    combined_umap = umap_model.fit_transform(combined_data)

    # Split back into cell and prototype embeddings
    cell_umap = combined_umap[: cells_tensor.shape[0], :]
    proto_umap = combined_umap[cells_tensor.shape[0] :, :]

    return cell_umap, proto_umap


def plot_joint_umap(
    cell_umap,
    cell_labels,
    batch_lables,
    protos=None,
    proto_size_base=50,
):
    """
    Visualize the UMAP embedding of cells and prototypes with:
    - A combined plot of cells and prototypes, colored by labels with a shared legend.
    - Prototypes' size adjusted based on confidence scores.

    Parameters:
    - cell_umap (np.ndarray): UMAP embedding of cells (n_cells, 2).
    - proto_umap (np.ndarray): UMAP embedding of prototypes (n_prototypes, 2).
    - cell_labels (list or np.ndarray): Labels assigned to cells (categorical).
    - proto_labels (list or np.ndarray): Labels assigned to prototypes (categorical).
    - proto_confidence (list or np.ndarray): Confidence scores for each prototype (continuous).
    - proto_size_base (int): Base size for prototypes; confidence scores adjust this size.
    """

    def add_legend(fig, ax, title, bbox_anchor):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=bbox_anchor,
            ncol=3,
            title=title,
        )

    def plot_ax(ax, cell_labels, label_name):
        # Combine labels for unified colors
        all_labels = np.unique(np.concatenate([cell_labels]))
        label_to_color_idx = {label: idx for idx, label in enumerate(all_labels)}
        color_map = plt.cm.get_cmap("tab20", len(all_labels))
        colors = ListedColormap(color_map(np.linspace(0, 1, len(all_labels))))

        if protos is not None:
            proto_umap, proto_labels, proto_confidence = protos
        # Initialize figure and axes

        def plot(umap, indices, label, s=10, zorder=1, **kwargs):
            if len(indices) == 0:
                return
            ax.scatter(
                umap[indices, 0],
                umap[indices, 1],
                label=label,
                color=colors(label_to_color_idx[label]),
                alpha=0.5,
                s=s,  # Fixed size for cells
                zorder=zorder,
                **kwargs,
            )

        # Plot cells and prototypes
        for label in all_labels:
            # Cells
            cell_indices = np.where(np.array(cell_labels) == label)[0]
            plot(cell_umap, cell_indices, label)

            # Prototypes
            if protos is not None:
                proto_indices = np.where(np.array(proto_labels) == label)[0]
                plot(
                    proto_umap,
                    proto_indices,
                    label,
                    edgecolor="k",
                    linewidth=0.5,
                    s=proto_size_base
                    * 2
                    * np.array(proto_confidence)[
                        proto_indices
                    ],  # Increased size for better visibility
                    alpha=0.8,
                    zorder=2,
                )

        # Titles and labels
        ax.set_title(f"UMAP of Cells and Prototypes (Colored by {label_name})")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plot_ax(axes[0], cell_labels, "cell types")
    add_legend(fig, axes[0], "Cell Types", (0.50, -0.1))

    plot_ax(axes[1], batch_lables, "studies")
    add_legend(fig, axes[1], "Studies", (0.85, -0.1))

    plt.tight_layout(rect=[0, 0.2, 1, 1])  # Increase the bottom margin for the legend
    plt.show()


def adata_umap(adata, n_neighbors=15, n_components=2, metric="cosine", use_pca=False):
    """
    Calculate UMAP embedding for the given AnnData object.

    Parameters:
        adata (AnnData): The annotated data matrix.
        n_neighbors (int): The size of the local neighborhood used for manifold approximation.
        n_components (int): The dimension of the space to embed into.
        metric (str): The distance metric to use for the neighborhood graph.

    Returns:
        AnnData: The input AnnData object with UMAP embedding added to `adata.obsm["X_umap"]`.
    """
    # Ensure PCA is run before computing the neighborhood graph
    if "X_pca" not in adata.obsm and use_pca:
        print("Running PCA...")
        sc.tl.pca(adata)

    # Compute the neighborhood graph
    print("Computing neighborhood graph...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric)

    # Compute UMAP
    print("Calculating UMAP embedding...")
    sc.tl.umap(adata, n_components=n_components)

    print("UMAP embedding added to `adata.obsm['X_umap']`.")
    return adata


def reconstruct_input(inputs, scpoli_encoder, recon_loss="nb"):
    """
    Reconstructs the input data from the decoder outputs using the negative binomial parameters.

    Args:
        dec_mean_gamma (torch.Tensor): Predicted mean (unscaled) from the decoder.
        sizefactor (torch.Tensor): Size factor for normalization (e.g., sequencing depth).
        combined_batch (torch.Tensor): One-hot encoded batch/condition information.
        n_conditions_combined (int): Number of combined conditions (for one-hot encoding).
        theta (torch.Tensor): Dispersion parameter weights.

    Returns:
        torch.Tensor: Reconstructed input data (approximated counts).
    """
    sizefactor, combined_batch, batch = (
        inputs["sizefactor"],
        inputs["combined_batch"],
        inputs["batch"],
    )
    z1, recon_loss, kl_div, mmd_loss = scpoli_encoder(**inputs)
    batch_embeddings = torch.hstack(
        [scpoli_encoder.embeddings[i](batch[:, i]) for i in range(batch.shape[1])]
    )
    outputs = scpoli_encoder.decoder(z1, batch_embeddings)

    if recon_loss == "nb":
        n_conditions_combined, theta = (
            scpoli_encoder.n_conditions_combined,
            scpoli_encoder.theta,
        )
        dec_mean_gamma, y1 = outputs

        # Expand size factors to match the shape of dec_mean_gamma
        size_factor_view = sizefactor.unsqueeze(1).expand(
            dec_mean_gamma.size(0), dec_mean_gamma.size(1)
        )

        # Calculate the normalized mean (mu) using size factors
        dec_mean = dec_mean_gamma * size_factor_view

        # Calculate dispersion parameter (theta) using the batch/condition info
        dispersion = F.linear(
            one_hot_encoder(combined_batch, n_conditions_combined), theta
        )
        dispersion = torch.exp(dispersion)  # Ensure dispersion is positive

        # Define the Negative Binomial distribution
        probs = dispersion / (dispersion + dec_mean)
        nb_dist = NegativeBinomial(total_count=dispersion, probs=probs)

        # Sample reconstructed gene expression data
        reconstructed_input = nb_dist.sample()
    else:
        recon_x, y1 = outputs
        reconstructed_input = torch.exp(recon_x) - 1
    return reconstructed_input


def calculate_umap(
    tensor,
    min_dist=0.5,
    spread=1.0,
    n_components=2,
    maxiter=None,
    alpha=1.0,
    gamma=1.0,
    negative_sample_rate=5,
    init_pos="spectral",
    random_state=0,
    metric="cosine",
):
    """
    Calculates UMAP for a given PyTorch tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (n_samples, n_features).
        Other parameters are UMAP configuration options.

    Returns:
        torch.Tensor: UMAP embeddings of shape (n_samples, n_components).
    """
    # Convert tensor to numpy array (UMAP operates on numpy)
    data = tensor.cpu().numpy()

    # Initialize UMAP with specified parameters
    umap_model = umap.UMAP(
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        n_epochs=maxiter,
        learning_rate=alpha,
        repulsion_strength=gamma,
        negative_sample_rate=negative_sample_rate,
        init=init_pos,
        random_state=random_state,
        metric=metric,
    )

    # Fit UMAP and transform data
    embeddings = umap_model.fit_transform(data)

    # Convert result back to a PyTorch tensor
    return embeddings


def model_reconstruction_plots(scpoli_cvae, adata, trainer, recon_loss="nb"):
    # recosntruct data
    # output umap and marker gene plots

    loader = trainer.prepare_scpoli_dataloader(adata, scpoli_cvae, shuffle=False)
    rec = []
    for data in loader:
        data = {key: data[key].to("cuda") for key in data}
        rec.append(reconstruct_input(data, scpoli_cvae, recon_loss))
    rec_tensor = torch.cat(rec, dim=0)
    rec_tensor = rec_tensor.detach().cpu()
    rec_adata = generate_proto_adata(
        rec_tensor, adata.obs.cell_type.values, adata.var.index.tolist()
    )
    u1 = plot_marker_gene_expressions(rec_adata)
    rec_umap = calculate_umap(rec_tensor)
    u2 = plot_joint_umap(rec_umap, adata.obs.cell_type, adata.obs.study)
    return u1, u2


import scanpy as sc
import pandas as pd


def perform_dge_analysis(
    adata,
    cell_type_1,
    cell_type_2,
    groupby_col="cell_type",
    method="wilcoxon",
    n_genes=20,
):
    """
    Perform differential gene expression analysis between two specific cell types.

    Parameters:
    - adata: AnnData object containing the data.
    - cell_type_1: The first cell type to compare (e.g., "CD4+ T cells").
    - cell_type_2: The second cell type to compare (e.g., "CD8+ T cells").
    - groupby_col: The column in `adata.obs` containing cell type annotations.
    - method: The method for DGE analysis (e.g., 'wilcoxon', 't-test', 'logreg').
    - n_genes: Number of top genes to display in the plot.

    Returns:
    - result_df: A pandas DataFrame containing the DGE results.
    """
    # Subset AnnData for the two specific cell types
    subset_adata = adata[adata.obs[groupby_col].isin([cell_type_1, cell_type_2])].copy()

    # Perform DGE analysis (one vs another)
    sc.tl.rank_genes_groups(
        subset_adata,
        groupby=groupby_col,
        method=method,
        groups=[cell_type_1],
        reference=cell_type_2,
    )

    # Visualize the results
    sc.pl.rank_genes_groups(subset_adata, n_genes=n_genes, sharey=False)

    # Extract results into a DataFrame
    result_df = sc.get.rank_genes_groups_df(subset_adata, group=cell_type_1)

    return result_df


import numpy as np

def assign_labels_based_on_similarity(similarity_matrix):
    """
    Assigns labels to samples based on the closest prototype in the similarity matrix.

    Parameters:
    - similarity_matrix (np.ndarray): A 2D array of shape (n_samples, n_prototypes),
      where each element represents the similarity between a sample and a prototype.

    Returns:
    - assigned_labels (np.ndarray): A 1D array of size n_samples, where each element
      is the label of the closest prototype.
    """
    similarity_matrix = similarity_matrix.detach().cpu().numpy()
    # Find the index of the prototype with the maximum similarity for each sample
    closest_prototype_indices = np.argmax(similarity_matrix, axis=1)

    # The index of the closest prototype corresponds to its label
    assigned_labels = closest_prototype_indices

    return assigned_labels


import scanpy as sc
import numpy as np
import pandas as pd

def preprocess_adata(adata):
    """
    Preprocesses an AnnData object by performing common checks and applying necessary preprocessing steps.
    Steps include:
        1. Missing values check and imputation
        2. Normalization to 10,000 total counts (if not already log-transformed)
        3. Log1p transformation (if not already applied)
        4. Identification of highly variable genes
        5. Data scaling (mean=0, variance=1)
        6. Filtering low-quality cells and genes

    Args:
        adata (AnnData): Input AnnData object

    Returns:
        AnnData: Preprocessed AnnData object
        pd.DataFrame: DataFrame log of actions taken
    """
    # Initialize a log list
    log = []

    # 1. Check for missing values
    if np.isnan(adata.X).any():
        log.append("Missing values found in adata.X. Filling with zeros.")
        adata.X = np.nan_to_num(adata.X)
    else:
        log.append("No missing values found in adata.X.")

    # 2. Check for normalization and log-transformation
    total_counts = adata.X.sum(axis=1)
    if not np.isclose(total_counts.mean(), 10000, atol=500):
        if np.max(adata.X) > 100:
            log.append("Counts are not normalized or log-transformed. Normalizing to 10,000.")
            sc.pp.normalize_total(adata, target_sum=10000)
        else:
            log.append("Data appears log-transformed. Skipping normalization.")
    else:
        log.append("Counts are already normalized.")

    # 3. Check for log transformation
    if (adata.X <= 0).any():
        log.append("Negative or zero values found. Skipping log transformation.")
    elif np.max(adata.X) > 100:
        log.append("Counts are not log-transformed. Applying log1p transformation.")
        sc.pp.log1p(adata)
    else:
        log.append("Counts are already log-transformed.")

    # 4. Identify highly variable genes
    if 'highly_variable' not in adata.var.columns:
        log.append("Highly variable genes not found. Calculating highly variable genes.")
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    else:
        log.append("Highly variable genes already identified.")

    # 5. Check for scaling
    if 'scaled' not in adata.uns:
        log.append("Data is not scaled. Scaling the data (mean=0, variance=1).")
        sc.pp.scale(adata, max_value=10)
        adata.uns['scaled'] = True
    else:
        log.append("Data is already scaled.")

    # 6. Filter low-quality cells and genes
    initial_cells = adata.n_obs
    initial_genes = adata.n_vars
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    filtered_cells = adata.n_obs
    filtered_genes = adata.n_vars
    log.append(f"Filtered low-quality data. Cells: {initial_cells} → {filtered_cells}, Genes: {initial_genes} → {filtered_genes}")

    # Return logs
    log.append("Preprocessing completed successfully.")
    log_df = pd.DataFrame(log, columns=["Action"])

    return adata, log_df

# Usage Example
# import scanpy as sc
# adata = sc.read_h5ad('your_file.h5ad')
# adata, preprocessing_log = preprocess_adata(adata)
# print(preprocessing_log)
