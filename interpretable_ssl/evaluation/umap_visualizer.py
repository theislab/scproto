from collections import Counter
from tqdm import tqdm
import torch
import umap
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class PrototypeVisualizer:
    def __init__(self, similarity_scores, embeddings, prototypes, obs, umaps=None):
        """
        Initialize the PrototypeVisualizer with required data.

        Parameters:
        - cell_umap (np.ndarray): UMAP embedding of cells (n_cells, 2).
        - cell_labels (list or np.ndarray): Labels assigned to cells (categorical).
        - umap_embedding (np.ndarray): UMAP embedding of prototypes (n_prototypes, 2).
        - prototype_labels (list or np.ndarray): Labels assigned to prototypes (categorical).
        - certainty_values (list or np.ndarray): Certainty values for each prototype (continuous).
        - samples_per_prototype (list or np.ndarray): Number of samples assigned to each prototype.
        """
        self.embeddings, self.prototypes = embeddings, prototypes
        self.similarity_scores = similarity_scores
        self.cell_labels = np.array(obs.cell_type)
        self.batches = np.array(obs.study)
        self.random_state = 42

        if umaps is None:
            self.cell_umap, self.prototype_umap = self.calculate_joint_umap()
        else:
            self.cell_umap, self.prototype_umap = umaps
        self.prototype_labels, self.certainty_values, self.samples_per_prototype = (
            self.assign_prototypes_with_similarity()
        )

        # Create a combined colormap for cells and prototypes
        all_labels = np.unique(
            np.concatenate([self.cell_labels, self.prototype_labels])
        )
        self.colors = ListedColormap(
            plt.cm.get_cmap("tab20", len(all_labels))(
                np.linspace(0, 1, len(all_labels))
            )
        )
        self.label_to_color_idx = {label: idx for idx, label in enumerate(all_labels)}

    def calculate_joint_umap(self, n_neighbors=5, min_dist=0.5):
        """
        Calculate joint UMAP for embeddings and prototypes and return separate UMAP embeddings.

        Parameters:
        - embeddings (np.ndarray or torch.Tensor): Tensor of shape (n_cells, n_features) for cell embeddings.
        - prototypes (np.ndarray or torch.Tensor): Tensor of shape (n_prototypes, n_features) for prototypes.
        - n_neighbors (int): Number of neighbors to consider in UMAP.
        - min_dist (float): Minimum distance between points in the UMAP embedding.
        - random_state (int): Random state for reproducibility.

        Returns:
        - cell_umap (np.ndarray): UMAP embedding for cells, shape (n_cells, 2).
        - prototype_umap (np.ndarray): UMAP embedding for prototypes, shape (n_prototypes, 2).
        """
        print("calculating umap")
        embeddings, prototypes = self.embeddings, self.prototypes
        random_state = self.random_state
        # Convert tensors to numpy arrays if necessary
        if isinstance(embeddings, np.ndarray):
            combined_data = np.vstack([embeddings, prototypes])
        else:
            combined_data = np.vstack(
                [embeddings.cpu().numpy(), prototypes.cpu().numpy()]
            )

        # Fit UMAP jointly on combined data
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric="cosine",
        )
        combined_umap = umap_model.fit_transform(combined_data)

        # Split combined UMAP embeddings into separate arrays
        n_cells = embeddings.shape[0]
        cell_umap = combined_umap[:n_cells]
        prototype_umap = combined_umap[n_cells:]
        print("done")
        return cell_umap, prototype_umap

    def assign_prototypes_with_similarity(self):
        """
        Assign embeddings to the most similar prototype based on similarity scores,
        compute majority label, certainty, and number of samples assigned.

        Parameters:
        - similarity_scores (torch.Tensor): Tensor of shape (n_samples, n_prototypes) with similarity scores.
        - cell_type_labels (pd.Series or np.ndarray or list): Array-like containing cell type labels (categorical).

        Returns:
        - prototype_labels (list): Majority label (original format) for each prototype.
        - certainty_values (list): Certainty for each prototype (percentage of embeddings with majority label).
        - samples_per_prototype (list): Number of samples assigned to each prototype.
        """
        similarity_scores, cell_type_labels = self.similarity_scores, self.cell_labels
        # Convert cell_type_labels to a Python list if needed
        if hasattr(cell_type_labels, "to_numpy"):  # If it's a Pandas Series or similar
            cell_type_labels = cell_type_labels.to_numpy().tolist()
        elif isinstance(cell_type_labels, torch.Tensor):
            cell_type_labels = cell_type_labels.tolist()

        # Assign each embedding to the prototype with the highest similarity
        closest_prototypes = torch.argmax(
            similarity_scores, dim=1
        )  # Shape: (n_samples,)

        # Ensure closest_prototypes is on the same device as similarity_scores
        closest_prototypes = closest_prototypes.to(similarity_scores.device)

        # Group embeddings by prototype
        prototype_labels = []
        certainty_values = []
        samples_per_prototype = []
        for proto_idx in tqdm(range(similarity_scores.shape[1])):
            # Find indices of embeddings assigned to the current prototype
            assigned_indices = (closest_prototypes == proto_idx).nonzero(as_tuple=True)[
                0
            ]

            # Extract the labels of these embeddings
            assigned_labels = [cell_type_labels[i] for i in assigned_indices.tolist()]

            # Store the number of samples assigned to the current prototype
            samples_per_prototype.append(len(assigned_labels))

            if len(assigned_labels) == 0:
                # No embeddings assigned to this prototype
                prototype_labels.append("none")
                certainty_values.append(0.0)
            else:
                # Find the majority label and its percentage
                label_counts = Counter(assigned_labels)
                majority_label, count = label_counts.most_common(1)[0]
                prototype_labels.append(majority_label)
                certainty_values.append(count / len(assigned_labels))

        return prototype_labels, certainty_values, samples_per_prototype

    def plot_cells_and_prototypes(self, ax, proto_size=25):
        """
        Plot cells and prototypes in a combined UMAP visualization, colored by labels.
        """
        for label in self.label_to_color_idx:
            cell_indices = np.where(self.cell_labels == label)[0]
            prototype_indices = [
                idx for idx, lbl in enumerate(self.prototype_labels) if lbl == label
            ]

            # Scatter prototypes
            if prototype_indices:
                ax.scatter(
                    self.prototype_umap[prototype_indices, 0],
                    self.prototype_umap[prototype_indices, 1],
                    color=self.colors(self.label_to_color_idx[label]),
                    edgecolor="k",
                    linewidth=0.5,
                    s=proto_size,
                    zorder=2,
                    alpha=0.7,
                )

            # Scatter cells
            ax.scatter(
                self.cell_umap[cell_indices, 0],
                self.cell_umap[cell_indices, 1],
                label=label,
                color=self.colors(self.label_to_color_idx[label]),
                alpha=0.7,
                s=10,
                zorder=1,
            )

        ax.set_title("UMAP of Cells and Prototypes (Colored by Labels)")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

    def plot_prototypes_colored(
        self, ax, values, title, colorbar_label, cmap, proto_size=25
    ):
        """
        Plot prototypes in UMAP space, colored by given values (e.g., certainty or sample count).
        """
        scatter = ax.scatter(
            self.prototype_umap[:, 0],
            self.prototype_umap[:, 1],
            c=values,
            cmap=cmap,
            s=proto_size,
            edgecolor="k",
            linewidth=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel("UMAP Dimension 1")
        ax.tick_params(axis="y", which="both", labelleft=False)

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label(colorbar_label)

    def visualize_old(self, proto_size=25, normalize_counts=True, log_scale=True):
        """
        Visualize the UMAP embedding of cells and prototypes with:
        - A combined plot of cells and prototypes, colored by labels with a shared legend.
        - A separate plot for prototypes colored by certainty values.
        - A separate plot for prototypes colored by the number of samples assigned.

        Parameters:
        - proto_size (int): Size of prototype points in the scatter plot.
        - normalize_counts (bool): Whether to normalize sample counts for visualization.
        - log_scale (bool): Whether to apply logarithmic scaling to sample counts.
        """
        # Normalize and/or log-transform samples_per_prototype
        samples_per_prototype = np.array(self.samples_per_prototype, dtype=np.float32)
        if log_scale:
            samples_per_prototype = np.log1p(
                samples_per_prototype
            )  # Handle zeros safely
        if normalize_counts:
            samples_per_prototype /= samples_per_prototype.max()

        # Initialize figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(28, 8), sharex=False, sharey=False)

        # Plot cells and prototypes combined
        self.plot_cells_and_prototypes(axes[0], proto_size)

        # Plot prototypes colored by certainty values
        self.plot_prototypes_colored(
            axes[1],
            self.certainty_values,
            title="UMAP of Prototypes (Colored by Certainty)",
            colorbar_label="Certainty",
            cmap="viridis",
            proto_size=proto_size,
        )

        # Plot prototypes colored by number of samples assigned
        self.plot_prototypes_colored(
            axes[2],
            samples_per_prototype,
            title="UMAP of Prototypes (Colored by Samples Assigned)",
            colorbar_label=(
                "Number of Samples Assigned (Log-Scaled)"
                if log_scale
                else "Number of Samples Assigned"
            ),
            cmap="plasma",
            proto_size=proto_size,
        )

        # Add a shared legend for cell and prototype labels
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=4,
            title="Cell/Prototype Types",
            fontsize="small",
        )

        plt.tight_layout(rect=[0, 0.2, 1, 1])
        plt.show()

    def visualize(self, proto_size=25, normalize_counts=True, log_scale=True):
        batch_labels = self.batches
        samples_per_prototype = self._normalize_samples(normalize_counts, log_scale)
        batch_labels_mapped = self._map_labels_to_indices(batch_labels)
        density = self._calculate_density(self.cell_umap)

        fig, axes = plt.subplots(1, 3, figsize=(28, 8), sharex=False, sharey=False)

        # Plot 1: Combined plot of cells and prototypes colored by cell type
        self._plot_labels_with_shared_colors(
            ax=axes[0],
            cell_labels=self.cell_labels,
            prototype_labels=self.prototype_labels,
            title="UMAP of Cells and Prototypes (Colored by Labels)",
            proto_size=proto_size,
            cmap="tab20",
            legend_title="Cell Types",
        )

        # Plot 2: Cells colored by batch, prototypes by certainty
        self._plot_labels_with_legend(
            ax=axes[1],
            cell_labels=batch_labels_mapped,
            prototype_labels=self.certainty_values,
            title="Cells Colored by Batch, Prototypes by Certainty",
            proto_size=proto_size,
            cmap="tab20",
            proto_cmap="viridis",
            legend_title="Batches",
        )

        # Plot 3: Cells colored by density, prototypes by sample count
        self._plot_labels_with_colorbar(
            ax=axes[2],
            cell_labels=density,
            prototype_labels=samples_per_prototype,
            title="Cells Colored by Density, Prototypes by Samples Assigned",
            proto_size=proto_size,
            cell_colorbar_label="Cell Density",
            proto_colorbar_label="Number of Samples Assigned",
            cmap="inferno",
            proto_cmap="plasma",
        )

        plt.tight_layout(rect=[0, 0.2, 1, 1])
        plt.show()

    def _plot_labels_with_shared_colors(
        self, ax, cell_labels, prototype_labels, title, proto_size, cmap, legend_title=None
    ):
        combined_labels = np.concatenate([cell_labels, prototype_labels])
        combined_labels_mapped = self._map_labels_to_indices(combined_labels)
        cell_labels_mapped = combined_labels_mapped[:len(cell_labels)]
        prototype_labels_mapped = combined_labels_mapped[len(cell_labels):]

        scatter_cells = ax.scatter(
            self.cell_umap[:, 0],
            self.cell_umap[:, 1],
            c=cell_labels_mapped,
            cmap=cmap,
            alpha=0.7,
            s=10,
            zorder=1,
        )

        scatter_prototypes = ax.scatter(
            self.prototype_umap[:, 0],
            self.prototype_umap[:, 1],
            c=prototype_labels_mapped,
            cmap=cmap,  # Ensure the same colormap
            edgecolor="k",
            linewidth=0.5,
            s=proto_size,
            zorder=2,
            alpha=0.7,
        )

        unique_labels = np.unique(combined_labels)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=scatter_cells.cmap(scatter_cells.norm(idx)), markersize=8)
            for idx, _ in enumerate(unique_labels)
        ]
        ax.legend(
            handles,
            unique_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            title=legend_title,
            fontsize="small",
            ncol=4,
        )

        ax.set_title(title)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

    def _plot_labels_with_legend(
        self, ax, cell_labels, prototype_labels, title, proto_size, cmap, proto_cmap=None, legend_title=None
    ):
        cell_labels_mapped = self._map_labels_to_indices(cell_labels)
        prototype_labels_mapped = (
            self._map_labels_to_indices(prototype_labels)
            if isinstance(prototype_labels[0], str)
            else prototype_labels
        )

        scatter_cells = ax.scatter(
            self.cell_umap[:, 0],
            self.cell_umap[:, 1],
            c=cell_labels_mapped,
            cmap=cmap,
            alpha=0.7,
            s=10,
            zorder=1,
        )

        scatter_prototypes = ax.scatter(
            self.prototype_umap[:, 0],
            self.prototype_umap[:, 1],
            c=prototype_labels_mapped,
            cmap=proto_cmap or cmap,
            edgecolor="k",
            linewidth=0.5,
            s=proto_size,
            zorder=2,
            alpha=0.7,
        )

        unique_labels = np.unique(cell_labels)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=scatter_cells.cmap(scatter_cells.norm(idx)), markersize=8)
            for idx, _ in enumerate(unique_labels)
        ]
        ax.legend(
            handles,
            unique_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            title=legend_title,
            fontsize="small",
            ncol=4,
        )

        ax.set_title(title)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

    def _plot_labels_with_colorbar(
        self, ax, cell_labels, prototype_labels, title, proto_size,
        cell_colorbar_label, proto_colorbar_label, cmap, proto_cmap=None
    ):
        scatter_cells = ax.scatter(
            self.cell_umap[:, 0],
            self.cell_umap[:, 1],
            c=cell_labels,
            cmap=cmap,
            alpha=0.7,
            s=10,
            zorder=1,
        )
        cbar_cells = plt.colorbar(scatter_cells, ax=ax, fraction=0.03, pad=0.08)
        cbar_cells.set_label(cell_colorbar_label)

        scatter_prototypes = ax.scatter(
            self.prototype_umap[:, 0],
            self.prototype_umap[:, 1],
            c=prototype_labels,
            cmap=proto_cmap or cmap,
            edgecolor="k",
            linewidth=0.5,
            s=proto_size,
            zorder=2,
            alpha=0.7,
        )
        cbar_prototypes = plt.colorbar(scatter_prototypes, ax=ax, fraction=0.03, pad=0.04)
        cbar_prototypes.set_label(proto_colorbar_label)

        ax.set_title(title)
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

    def _normalize_samples(self, normalize_counts, log_scale):
        samples = np.array(self.samples_per_prototype, dtype=np.float32)
        if log_scale:
            samples = np.log1p(samples)
        if normalize_counts:
            samples /= samples.max()
        return samples

    def _calculate_density(self, data):
        from scipy.stats import gaussian_kde
        return gaussian_kde(data.T)(data.T)

    def _map_labels_to_indices(self, labels):
        unique_labels = np.unique(labels)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        return np.array([label_to_index[label] for label in labels])
