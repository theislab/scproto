import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

import torch


def plot_marker_gene_expressions(
    adata,
    cell_types_to_filter=["CD4+ T cells", "CD8+ T cells"],
    x_gene="CD4",
    y_gene="CD8A",
    cell_type_column="cell_type",
):
    """
    Plot CD8A vs TYROBP expression colored by density and cell type.

    Args:
        adata (AnnData): The input AnnData object containing gene expression data.
        cell_types_to_filter (list): List of cell types to include in the plot.
        x_gene (str): The name of the gene to plot on the x-axis.
        y_gene (str): The name of the gene to plot on the y-axis.
        cell_type_column (str): Column name in adata.obs specifying cell types.

    Returns:
        None
    """
    # Step 1: Filter cells based on cell type
    adata_filtered = adata[adata.obs[cell_type_column].isin(cell_types_to_filter)]
    print(len(adata_filtered))
    print(len(adata_filtered.var))
    # Convert sparse matrix to dense if necessary, then flatten
    x = adata_filtered[:, x_gene].X
    y = adata_filtered[:, y_gene].X

    # Check if x and y are sparse, convert to dense if needed
    if hasattr(x, "toarray"):
        x = x.toarray()
    if hasattr(y, "toarray"):
        y = y.toarray()

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()

    # Step 3: Get cell type information
    cell_types = adata_filtered.obs[cell_type_column]

    # Step 4: Compute density for density-colored scatter plot
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)

    # Step 5: Create the plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

    # Plot 1: Density-colored scatter plot
    axes[0].scatter(x, y, c=density, cmap="viridis", s=5, alpha=0.8)
    axes[0].set_title(f"{y_gene} vs {x_gene} (Density Colored)")
    axes[0].set_xlabel(f"{x_gene} Expression")
    axes[0].set_ylabel(f"{y_gene} Expression")
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap="viridis"), ax=axes[0], label="Density"
    )

    # Plot 2: Cell type-colored scatter plot
    unique_cell_types = cell_types_to_filter
    palette = sns.color_palette("hsv", len(unique_cell_types))
    color_dict = dict(zip(unique_cell_types, palette))
    for cell_type in unique_cell_types:
        idx = cell_types == cell_type
        axes[1].scatter(
            x[idx],
            y[idx],
            c=[color_dict[cell_type]] * sum(idx),
            label=cell_type,
            s=5,
            alpha=0.8,
        )
    axes[1].set_title(f"{y_gene} vs {x_gene} (Cell Type Colored)")
    axes[1].set_xlabel(f"{x_gene} Expression")
    axes[1].set_ylabel(f"{y_gene} Expression")
    axes[1].legend(title="Cell Type", loc="best", markerscale=3)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    return plt


import numpy as np
import pandas as pd
from collections import Counter
import anndata


def assign_prototype_labels(adata, similarity_tensor, cell_type_column="cell_type"):
    """
    Assigns each sample in AnnData to the prototype with the highest similarity,
    calculates the majority cell type for each prototype, and assigns the prototype label.

    Args:
        adata (AnnData): The AnnData object containing `obs[cell_type_column]` labels.
        similarity_tensor (torch.Tensor or np.ndarray): A tensor of shape (n_samples, n_prototypes)
                                                        representing the similarity of each sample to each prototype.
        cell_type_column (str): The column in `adata.obs` that contains cell type labels.

    Returns:
        AnnData: Updated AnnData object with prototype labels and confidence.
    """
    # Ensure similarity_tensor is a numpy array
    if isinstance(similarity_tensor, torch.Tensor):
        similarity_tensor = similarity_tensor.cpu().numpy()

    # Step 1: Assign each sample to the prototype with the highest similarity
    prototype_assignments = np.argmax(similarity_tensor, axis=1)

    # Step 2: Add the prototype assignments to `adata.obs`
    adata.obs["prototype"] = prototype_assignments

    # Step 3: Calculate majority cell type and confidence for each prototype
    prototype_labels = []
    prototype_confidences = []

    for prototype in range(similarity_tensor.shape[1]):
        # Get samples assigned to this prototype
        assigned_samples = adata.obs[adata.obs["prototype"] == prototype]

        # Get the cell types of these samples
        cell_types = assigned_samples[cell_type_column]

        if len(cell_types) > 0:
            # Count the frequency of each cell type
            cell_type_counts = Counter(cell_types)

            # Determine the majority cell type and its confidence
            majority_cell_type, majority_count = cell_type_counts.most_common(1)[0]
            confidence = majority_count / len(cell_types)
        else:
            # Handle prototypes with no assigned samples
            majority_cell_type = "Unknown"
            confidence = 0.0

        prototype_labels.append(majority_cell_type)
        prototype_confidences.append(confidence)

    # Step 4: Create a DataFrame for prototype labels and confidences
    prototype_df = pd.DataFrame(
        {
            "prototype": range(similarity_tensor.shape[1]),
            "prototype_label": prototype_labels,
            "prototype_confidence": prototype_confidences,
        }
    )

    return prototype_df


def generate_proto_adata(x, cell_types, gene_panel=None, confidence=None):
    """
    Generate an AnnData object for prototypes from input data.

    Args:
        x (np.ndarray or torch.Tensor): A 2D array or tensor of shape (n_prototypes, n_features),
                                        where each row represents a prototype's data.
        cell_types (list): A list of cell type labels for each prototype.
        gene_panel (list): A list of gene names corresponding to the features (columns) in x.

    Returns:
        AnnData: A new AnnData object containing prototype data, cell types, and gene panel.
    """
    # Convert x to numpy if it's a PyTorch tensor
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    # Validate input dimensions
    assert (
        len(cell_types) == x.shape[0]
    ), "Number of cell types must match the number of prototypes."

    # Create an AnnData object
    proto_adata = anndata.AnnData(X=x)

    # Add cell types to proto_adata.obs
    proto_adata.obs["cell_type"] = pd.Categorical(cell_types)

    if confidence is not None:
        proto_adata.obs["confidence"] = confidence
    # Add gene panel to proto_adata.var
    if gene_panel is not None:
        proto_adata.var.index = pd.Index(gene_panel)

    return proto_adata
