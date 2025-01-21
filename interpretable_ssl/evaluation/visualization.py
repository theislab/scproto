import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scanpy as sc
from scipy.stats import gaussian_kde

random_state = 42


def calculate_umap(embeddings, prototypes=None, metric="cosine"):
    num_cells = embeddings.shape[0]

    # Convert embeddings to numpy arrays if they are tensors
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()

    if prototypes is not None:
        num_prototypes = len(prototypes)

        # Convert prototypes to numpy arrays if they are tensors
        if torch.is_tensor(prototypes):
            prototypes = prototypes.detach().cpu().numpy()

        # Combine embeddings and prototypes for UMAP
        combined_data = np.vstack((embeddings, prototypes))
        combined_adata = sc.AnnData(combined_data)
        combined_adata.obs["type"] = ["cell"] * num_cells + [
            "prototype"
        ] * num_prototypes

        # # Check for cells with all zero counts
        # all_zero_cells = np.all(combined_adata.X == 0, axis=1)
        # if np.any(all_zero_cells):
        #     print(f"Number of all-zero cells: {np.sum(all_zero_cells)}")
        #     combined_adata = combined_adata[~all_zero_cells]  # Remove all-zero cells

        # # Normalize the data
        # sc.pp.normalize_total(combined_adata, target_sum=1e4)

        # # Check for NaNs after normalization
        # if np.any(np.isnan(combined_adata.X)):
        #     print("Data contains NaNs after normalization")

        # # Log transform the data
        # sc.pp.log1p(combined_adata)

        # # Check for NaNs after log transformation
        # if np.any(np.isnan(combined_adata.X)):
        #     print("Data contains NaNs after log transformation")

        # # Scale the data
        # sc.pp.scale(combined_adata, max_value=10)

        # # Check for NaNs after scaling
        # if np.any(np.isnan(combined_adata.X)):
        #     print("Data contains NaNs after scaling")

        # Perform UMAP on the combined data with specified metric
        sc.pp.neighbors(
            combined_adata, use_rep="X", metric=metric, random_state=random_state
        )
        # sc.pp.neighbors(combined_adata, use_rep='X', n_neighbors=15)
        sc.tl.umap(combined_adata, random_state=random_state)

        # Extract UMAP embeddings
        umap_embedding = combined_adata.obsm["X_umap"]
        cell_umap = umap_embedding[:num_cells]
        prototype_umap = umap_embedding[num_cells:]
    else:
        # Perform UMAP on the embeddings only with specified metric
        adata = sc.AnnData(embeddings)
        sc.pp.neighbors(adata, use_rep="X", metric=metric, random_state=random_state)
        sc.tl.umap(adata, random_state=random_state)

        # Extract UMAP embeddings
        umap_embedding = adata.obsm["X_umap"]
        cell_umap = umap_embedding
        prototype_umap = None

    return cell_umap, prototype_umap


def plot_umap(
    cell_umap,
    prototype_umap,
    cell_types,
    study_labels,
    augmentation_labels=None,
    save_plot=True,
    save_path=None,
):
    # Determine the number of subplots based on whether augmentation labels are provided
    n_plots = 3 if augmentation_labels is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(20, 8))

    # Optionally plot cell embeddings colored by augmentation
    if augmentation_labels is not None:
        unique_augmentations = np.unique(augmentation_labels)
        unique_colors_augmentations = plt.cm.get_cmap(
            "tab20", len(unique_augmentations)
        )
        colors_augmentations = ListedColormap(
            unique_colors_augmentations(np.linspace(0, 1, len(unique_augmentations)))
        )

        for i, aug in enumerate(unique_augmentations):
            indices = np.where(augmentation_labels == aug)[0]
            axes[0].scatter(
                cell_umap[indices, 0],
                cell_umap[indices, 1],
                label=aug,
                color=colors_augmentations(i),
                alpha=0.6,
                s=20,
            )

        if prototype_umap is not None:
            axes[0].scatter(
                prototype_umap[:, 0],
                prototype_umap[:, 1],
                color="white",
                edgecolor="black",
                s=100,
                marker="o",
                label="Prototypes",
            )

        axes[0].set_title("UMAP of Cell Embeddings with Augmentations Highlighted")
        axes[0].set_xlabel("UMAP Dimension 1")
        axes[0].set_ylabel("UMAP Dimension 2")

    # Plot cell embeddings colored by cell type
    unique_cell_types = np.unique(cell_types)
    unique_colors = plt.cm.get_cmap("tab20", len(unique_cell_types))
    colors = ListedColormap(unique_colors(np.linspace(0, 1, len(unique_cell_types))))

    for i, cell_type in enumerate(unique_cell_types):
        indices = np.where(cell_types == cell_type)[0]
        axes[1 if augmentation_labels is not None else 0].scatter(
            cell_umap[indices, 0],
            cell_umap[indices, 1],
            label=cell_type,
            color=colors(i),
            alpha=0.6,
            s=20,
        )

    if prototype_umap is not None:
        axes[1 if augmentation_labels is not None else 0].scatter(
            prototype_umap[:, 0],
            prototype_umap[:, 1],
            color="white",
            edgecolor="black",
            s=100,
            marker="o",
            label="Prototypes",
        )

    axes[1 if augmentation_labels is not None else 0].set_title(
        "UMAP of Cell Embeddings with Cell Types Highlighted"
    )
    axes[1 if augmentation_labels is not None else 0].set_xlabel("UMAP Dimension 1")
    axes[1 if augmentation_labels is not None else 0].set_ylabel("UMAP Dimension 2")

    # Plot cell embeddings colored by study
    unique_studies = np.unique(study_labels)
    unique_colors_studies = plt.cm.get_cmap("tab20", len(unique_studies))
    colors_studies = ListedColormap(
        unique_colors_studies(np.linspace(0, 1, len(unique_studies)))
    )

    for i, study in enumerate(unique_studies):
        indices = np.where(study_labels == study)[0]
        axes[2 if augmentation_labels is not None else 1].scatter(
            cell_umap[indices, 0],
            cell_umap[indices, 1],
            label=study,
            color=colors_studies(i),
            alpha=0.6,
            s=20,
        )

    if prototype_umap is not None:
        axes[2 if augmentation_labels is not None else 1].scatter(
            prototype_umap[:, 0],
            prototype_umap[:, 1],
            color="white",
            edgecolor="black",
            s=100,
            marker="o",
            label="Prototypes",
        )

    axes[2 if augmentation_labels is not None else 1].set_title(
        "UMAP of Cell Embeddings with Studies Highlighted"
    )
    axes[2 if augmentation_labels is not None else 1].set_xlabel("UMAP Dimension 1")
    axes[2 if augmentation_labels is not None else 1].set_ylabel("UMAP Dimension 2")

    # Adjust layout to make space for the legends
    plt.tight_layout(rect=[0, 0.2, 1, 1])

    # Adding legends below the plots
    if augmentation_labels is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.20, -0.1),
            ncol=3,
            title="Augmentations",
        )

    handles, labels = axes[
        1 if augmentation_labels is not None else 0
    ].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.50 if n_plots == 3 else 0.25, -0.1),
        ncol=3,
        title="Cell Types",
    )

    handles, labels = axes[
        2 if augmentation_labels is not None else 1
    ].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.80 if n_plots == 3 else 0.75, -0.1),
        ncol=3,
        title="Studies",
    )

    if save_plot:
        plt.savefig(f"{save_path}/ref-umap.png", bbox_inches="tight")
    else:
        plt.show()


from scipy.stats import gaussian_kde

import matplotlib.colors as mcolors


def plot_3umaps(
    cell_umap,
    prototype_umap,
    cell_types,
    study_labels,
    prototype_labels=None,
    save_plot=True,
    save_path_list=None,
):

    def plot_scatter(
        ax,
        data_umap,
        labels=None,
        label_title=None,
        prototypes=None,
        prototype_labels=None,
        exclude_prototypes=False,
        color_by_density=False,
    ):
        if color_by_density:
            # Calculate point density
            xy = np.vstack([data_umap[:, 0], data_umap[:, 1]])
            density = gaussian_kde(xy)(xy)
            scatter = ax.scatter(
                data_umap[:, 0],
                data_umap[:, 1],
                c=density,
                cmap="viridis",
                s=20,
                alpha=0.6,
            )
            cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
            cbar.set_label("Density")
        else:
            unique_labels = np.unique(labels)
            unique_colors = plt.cm.get_cmap("tab20", len(unique_labels))

            colors = ListedColormap(
                unique_colors(np.linspace(0, 1, len(unique_labels)))
            )

            for i, label in enumerate(unique_labels):
                indices = np.where(labels == label)[0]
                ax.scatter(
                    data_umap[indices, 0],
                    data_umap[indices, 1],
                    label=label,
                    color=colors(i),
                    alpha=0.6,
                    s=20,
                )

                if prototypes is not None and not exclude_prototypes:
                    prototype_indices = [
                        idx for idx, lbl in enumerate(prototype_labels) if lbl == label
                    ]
                    if prototype_indices:

                        ax.scatter(
                            prototypes[prototype_indices, 0],
                            prototypes[prototype_indices, 1],
                            color=colors(i),
                            edgecolor="black",
                            linewidth=0.5,
                            s=70,
                            zorder=2,
                            alpha=0.8,
                            # label="Prototypes",
                        )

            # Plot prototypes with labels not in cell types
            if prototypes is not None:
                extra_indices = [
                    idx
                    for idx, lbl in enumerate(prototype_labels)
                    if lbl not in unique_labels
                ]
                if extra_indices:
                    ax.scatter(
                        prototypes[extra_indices, 0],
                        prototypes[extra_indices, 1],
                        color="white",  # Use a distinct color for extra prototypes
                        edgecolor="black",
                        linewidth=0.5,
                        s=70,
                        zorder=3,
                        alpha=0.9,
                        label="Extra Prototypes",  # Add a legend for extra prototypes
                    )
        if label_title:
            ax.set_title(f"UMAP of Cell Embeddings with {label_title} Highlighted")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

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

    # Determine the number of subplots
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot cell embeddings colored by density (without prototypes)
    plot_scatter(axes[0], cell_umap, color_by_density=True)
    axes[0].set_title("UMAP of Cell Embeddings Colored by Density")

    # Plot cell embeddings colored by cell types (with prototypes)
    plot_scatter(
        axes[1], cell_umap, cell_types, "Cell Types", prototype_umap, prototype_labels
    )
    if np.unique(cell_types).shape[0] < 50:
        add_legend(fig, axes[1], "Cell Types", (0.50, -0.1))

    # Plot cell embeddings colored by studies (without prototypes)
    plot_scatter(axes[2], cell_umap, study_labels, "Studies", exclude_prototypes=True)
    add_legend(fig, axes[2], "Studies", (0.85, -0.1))

    plt.tight_layout(rect=[0, 0.2, 1, 1])

    if save_path_list is not None:
        for save_path in save_path_list:
            fig.savefig(
                save_path, bbox_inches="tight", pad_inches=0.5
            )  # Increase pad_inches as needed

    return fig


def plot_3umaps2(
    cell_umap,
    prototype_umap,
    cell_types,
    study_labels,
    proto_labels,
    proto_confidence=None,
    save_plot=True,
    save_path_list=None,
    proto_size_base=100,
):

    def plot_scatter(
        ax,
        data_umap,
        labels=None,
        label_title=None,
        prototypes=None,
        proto_labels=None,
        proto_confidence=None,
        exclude_prototypes=False,
        color_by_density=False,
        label_to_color_idx=None,
        shared_colors=None,
    ):
        if color_by_density:
            # Calculate point density
            xy = np.vstack([data_umap[:, 0], data_umap[:, 1]])
            density = gaussian_kde(xy)(xy)
            scatter = ax.scatter(
                data_umap[:, 0],
                data_umap[:, 1],
                c=density,
                cmap="viridis",
                s=20,
                alpha=0.6,
            )
            cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
            cbar.set_label("Density")
        else:
            for label, idx in label_to_color_idx.items():
                indices = np.where(np.array(labels) == label)[0]
                ax.scatter(
                    data_umap[indices, 0],
                    data_umap[indices, 1],
                    label=label,
                    color=shared_colors(idx),
                    alpha=0.6,
                    s=20,
                )

        # Visualize prototypes
        if prototypes is not None and not exclude_prototypes:
            visualize_prototypes(
                ax,
                prototypes,
                proto_labels,
                proto_confidence,
                proto_size_base,
                label_to_color_idx,
                shared_colors,
            )

        if label_title:
            ax.set_title(f"UMAP of Cell Embeddings with {label_title} Highlighted")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

    def visualize_prototypes(
        ax,
        prototypes,
        proto_labels,
        proto_confidence,
        proto_size_base,
        label_to_color_idx,
        shared_colors,
    ):
        """
        Visualizes prototypes on the UMAP plot.
        - Prototypes are colored by their labels and sized by confidence.
        """
        if proto_confidence is None:
            proto_confidence = np.ones(prototypes.shape[0])  # Default confidence

        for label, idx in label_to_color_idx.items():
            indices = np.where(np.array(proto_labels) == label)[0]
            ax.scatter(
                prototypes[indices, 0],
                prototypes[indices, 1],
                color=shared_colors(idx),
                edgecolor="k",
                linewidth=0.5,
                s=proto_size_base
                * proto_confidence[indices],  # Scale size by confidence
                alpha=0.8,
                label=f"Prototype: {label}",
            )

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

    # Combine cell types and prototype labels for a shared colormap
    all_labels = np.unique(np.concatenate([cell_types, proto_labels]))
    label_to_color_idx = {label: idx for idx, label in enumerate(all_labels)}
    shared_colors = plt.cm.get_cmap("tab20", len(all_labels))

    # Determine the number of subplots
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot cell embeddings colored by density (without prototypes)
    plot_scatter(axes[0], cell_umap, color_by_density=True)
    axes[0].set_title("UMAP of Cell Embeddings Colored by Density")

    # Plot cell embeddings colored by cell types (with prototypes)
    plot_scatter(
        axes[1],
        cell_umap,
        cell_types,
        "Cell Types",
        prototypes=prototype_umap,
        proto_labels=proto_labels,
        proto_confidence=proto_confidence,
        label_to_color_idx=label_to_color_idx,
        shared_colors=shared_colors,
    )
    add_legend(fig, axes[1], "Cell Types", (0.50, -0.1))

    # Plot cell embeddings colored by studies (without prototypes)
    plot_scatter(
        axes[2],
        cell_umap,
        study_labels,
        "Studies",
        exclude_prototypes=True,
        label_to_color_idx=label_to_color_idx,
        shared_colors=shared_colors,
    )
    add_legend(fig, axes[2], "Studies", (0.85, -0.1))

    plt.tight_layout(rect=[0, 0.2, 1, 1])

    if save_plot and save_path_list is not None:
        for save_path in save_path_list:
            fig.savefig(
                save_path, bbox_inches="tight", pad_inches=0.5
            )  # Increase pad_inches as needed

    return fig


import matplotlib.pyplot as plt

def plot_f1_scores_per_class(f1_scores_dict, class_names):
    """
    Plots a grouped bar plot of F1 scores per class for multiple splits, with values displayed on top of the bars.

    Parameters:
    - f1_scores_dict: Dictionary where keys are split names (e.g., 'Train', 'Validation', 'Test')
      and values are lists of per-class F1 scores.
      Example: {
          'Train': [0.8, 0.7, 0.9],
          'Validation': [0.75, 0.65, 0.85],
          'Test': [0.78, 0.68, 0.88]
      }
    - class_names: List of class names corresponding to the F1 scores.
    """
    # Number of classes and splits
    num_classes = len(class_names)
    num_splits = len(f1_scores_dict)
    bar_width = 0.25
    indices = range(num_classes)

    # Create the plot
    plt.figure(figsize=(12, 6))
    for i, (split_name, f1_scores) in enumerate(f1_scores_dict.items()):
        bar_positions = [x + i * bar_width for x in indices]
        bars = plt.bar(
            bar_positions,
            f1_scores,
            width=bar_width,
            label=split_name,
        )

        # Add values on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Formatting
    plt.xticks(
        [x + bar_width * (num_splits / 2 - 0.5) for x in indices],
        class_names,
        rotation=45,
        ha="right",
    )
    plt.xlabel("Class Names")
    plt.ylabel("F1 Score")
    plt.title("F1 Score comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

