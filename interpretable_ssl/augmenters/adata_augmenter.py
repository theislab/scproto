from scarches.dataset.scpoli.anndata import MultiConditionAnnotatedDataset
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import scipy.stats as stats
from sklearn.decomposition import PCA
import logging
import random

from torch.utils.data import get_worker_info
from interpretable_ssl.augmenters.graph_handler import GraphHandler
import scipy.sparse as sp
import bbknn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiCropsDataset(MultiConditionAnnotatedDataset):
    def __init__(
        self,
        adata,
        original_indicies,
        n_augmentations,
        augmentation_type="cell_type",
        k_neighbors=10,  # seacell use 50
        longest_path=3,
        dimensionality_reduction=None,
        n_components=50,
        supervised_ratio=0.1,
        use_bknn=0,
        knn_similarity='euclidean',
        **kwargs,
    ):
        """
        Initialize the augmented dataset handler for scPoli model and trainer.

        Parameters
        ----------
        adata : `~anndata.AnnData`
            Annotated data matrix.
        n_augmentations : int
            Number of augmentations to perform for each cell in a batch.
        augmentation_type : str
            Type of augmentation to use ("knn", "cell_type", "scanpy-knn", or "negative_binomial").
        k_neighbors : int
            Number of neighbors for kNN graph (only used if augmentation_type is "knn" or "scanpy-knn").
        longest_path : int
            Maximum length of the random walk path.
        dimensity_reduction : str or None
            Type of dimensionality reduction to apply ("pca" or None).
        n_components : int or None
            Number of principal components to use for PCA if dimensity_reduction is "pca".
        kwargs : dict
            Additional arguments for the parent class.
        """
        self.n_augmentations = n_augmentations
        self.augmentation_type = augmentation_type
        self.adata = adata
        self.k_neighbors = k_neighbors
        self.longest_path = longest_path
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        self.knn_graph = None

        self.graph_handler = GraphHandler(original_indicies)
        self.supervised_ratio = supervised_ratio
        self.use_bknn = use_bknn
        self.knn_similarity = knn_similarity
        if self.augmentation_type not in ["cell_type", "nb"]:
            self.set_graph()
        super().__init__(adata, **kwargs)

    def get_random_indices(self, n, specific_index):
        """
        Returns a list of n random indices including a specific index.

        Args:
            dataset_size (int): Total number of samples in the dataset.
            n (int): Number of random indices to return.
            specific_index (int): The specific index that must be included.

        Returns:
            list: A list of n random indices, including the specific index.
        """
        dataset_size = len(self.adata)
        # Ensure n is valid
        if n > dataset_size:
            raise ValueError("n cannot be greater than the dataset size")

        # Get a random sample of indices excluding the specific index
        indices = random.sample(
            [i for i in range(dataset_size) if i != specific_index], n - 1
        )

        # Append the specific index to the list
        indices.append(specific_index)
        return indices

    def build_knn_graph(self):
        logger.info(f"building knn graph")
        graph = self.build_scanpy_knn_graph()
        return graph

    def _build_knn_graph_with_sklearn(self, indices=None):
        """
        Build a k-nearest neighbors graph based on the expression profiles using sklearn's NearestNeighbors.
        """
        if indices is None:
            X = self.adata.X
        else:
            X = self.adata[indices].X
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
        else:
            worker_id = None
        logger.info(
            f"wid: {worker_id} Constructing NearestNeighbors with k={self.k_neighbors + 1} on x with shape: {X.shape}"
        )

        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm="auto").fit(
            X
        )
        distances, indices = nbrs.kneighbors(X)

        logger.debug("NearestNeighbors fitting completed.")

        # Exclude the first neighbor for each node, which is the node itself
        indices = indices[:, 1:]
        distances = distances[:, 1:]

        logger.info("k-nearest neighbors graph construction completed.")
        logger.debug(
            f"indices shape: {indices.shape}, distances shape: {distances.shape}"
        )

        return indices, distances

    def _apply_pca_if_needed(self):
        """
        Apply PCA to the data if dimensionality reduction is set to 'pca'.
        """
        if self.dimensionality_reduction == "pca":
            if self.n_components is None:
                logger.error("n_components must be specified when using PCA.")
                raise ValueError("n_components must be specified when using PCA.")

            logger.info(f"Performing PCA with n_components={self.n_components}.")
            sc.tl.pca(self.adata, n_comps=self.n_components)
            logger.debug(
                f"PCA completed. Shape of data after PCA: {self.adata.obsm['X_pca'].shape}"
            )

    def _build_knn_graph_with_scanpy(self):
        """
        Build the k-nearest neighbors graph using Scanpy.
        """
        logger.info(f"Running Scanpy neighbors with k={self.k_neighbors + 1}.")
        if self.use_bknn:
            logger.info('print generating bknn graph')
            num_batches = self.adata.obs['study'].nunique()
            neighbors_within_batch = int(self.k_neighbors / num_batches)
            bbknn.bbknn(self.adata, batch_key='study', neighbors_within_batch=neighbors_within_batch)
        else:
            sc.pp.neighbors(
                self.adata,
                n_neighbors=self.k_neighbors + 1,
                metric=self.knn_similarity,
                # method='faiss',
                use_rep="X_pca" if self.dimensionality_reduction == "pca" else None,
            )
            
        # # Retrieve the adjacency matrix (kNN graph) created by Scanpy
        # adjacency_matrix = self.adata.obsp["connectivities"]
        
        # # Retrieve the study labels
        # study_labels = self.adata.obs["study"].values  # Replace "study" with the actual column name
        
        # # Create a mask to filter edges
        # row, col = adjacency_matrix.nonzero()  # Get the non-zero indices (edges)
        # valid_edges = study_labels[row] == study_labels[col]  # True for edges within the same study
        
        # # Filter the adjacency matrix
        # filtered_adj = sp.csr_matrix(
        #     (adjacency_matrix.data[valid_edges], (row[valid_edges], col[valid_edges])),
        #     shape=adjacency_matrix.shape,
        # )
        
        # # Replace the original adjacency matrix with the filtered one
        # self.adata.obsp["connectivities"] = filtered_adj

    def _extract_knn_graph_info(self):
        """
        Extract indices and distances from the kNN graph.
        """
        knn_graph = self.adata.obsp["distances"]
        logger.debug("kNN graph extracted from Scanpy.")

        indices = []
        distances = []

        logger.info("Processing the kNN graph to extract indices and distances.")

        for i in range(knn_graph.shape[0]):
            nonzero_indices = knn_graph[i].nonzero()[1]
            nonzero_distances = knn_graph[i, nonzero_indices].toarray().flatten()

            indices.append(nonzero_indices)
            distances.append(nonzero_distances)

        logger.info("Finished processing the kNN graph.")

        indices = np.array(indices)
        distances = np.array(distances)

        logger.info(
            f"indices shape: {indices.shape}, distances shape: {distances.shape}"
        )

        return indices, distances

    def build_scanpy_knn_graph(self):
        """
        Build a k-nearest neighbors graph using scanpy.pp.neighbors and format it
        to match the output of build_knn_graph, with the option to apply PCA.
        """
        logger.info("Starting to build k-nearest neighbors graph using Scanpy.")
        self._apply_pca_if_needed()
        self._build_knn_graph_with_scanpy()
        return self._extract_knn_graph_info()

    def build_community_graph(self):
        """
        Build a k-nearest neighbors graph, perform Leiden community detection,
        and assign community labels to the cells.
        """
        logger.info("Starting to build community graph.")
        self._apply_pca_if_needed()
        self._build_knn_graph_with_scanpy()

        logger.info("Performing Leiden community detection.")
        sc.tl.leiden(self.adata, key_added="leiden_community")

        logger.debug("Leiden community detection completed.")
        self.adata.obs["leiden_community"] = self.adata.obs["leiden_community"].astype(
            "category"
        )
        logger.info("Community labels assigned to the cells.")

    def random_walk(self, start_index):
        """
        Perform a random walk on the kNN graph starting from the given index.
        """

        current_index = start_index
        path_length = np.random.randint(1, self.longest_path + 1)
        indices, distances = self.knn_graph

        for _ in range(path_length):
            neighbors = indices[current_index]
            weights = distances[current_index]
            weights = weights / weights.sum()  # Normalize weights
            next_index = np.random.choice(neighbors, p=weights)
            current_index = next_index

        return current_index

    def set_graph(self):
        if self.knn_graph is None:
            self.knn_graph = self.build_knn_graph()

    def knn_augment(self, index):
        self.set_graph()
        augmented_indices = [index]  # Start with the original index
        for _ in range(self.n_augmentations - 1):
            augmented_indices.append(self.random_walk(index))
        return augmented_indices

    def community_augment(self, index):
        """
        Augment the data by choosing other random cells from the same community.
        """
        if "leiden_community" not in self.adata.obs:
            self.build_community_graph()
        # Get the community label for the current cell
        community_label = self.adata.obs.iloc[index]["leiden_community"]

        # Get all indices of cells in the same community
        same_community_indices = np.where(
            self.adata.obs["leiden_community"] == community_label
        )[0]

        # Sample n_augmentations - 1 indices from the same community
        sampled_indices = np.random.choice(
            same_community_indices, self.n_augmentations - 1, replace=False
        )

        # Include the original index
        augmented_indices = [index] + sampled_indices.tolist()

        return augmented_indices

    def cell_type_augment(self, index):
        cell_type = self.adata.obs.iloc[index]["cell_type"]

        # Apply batch filtering
        batch = self.adata.obs.iloc[index]["study"]
        same_type_diff_batch_indices = np.where(
            (self.adata.obs["cell_type"] == cell_type)
            & (self.adata.obs["study"] != batch)
        )[0]

        # Check if batch filtering leaves more than n samples
        if len(same_type_diff_batch_indices) > self.n_augmentations:
            # Use the batch-filtered indices
            final_indices = same_type_diff_batch_indices
        else:
            # Fall back to using indices with only cell type filtering
            final_indices = np.where(self.adata.obs["cell_type"] == cell_type)[0]
        augmented_indices = [index]  # Start with the original index
        sampled_indices = np.random.choice(
            final_indices, self.n_augmentations - 1, replace=False
        )
        augmented_indices.extend(sampled_indices)
        return augmented_indices

    def negative_binomial_augment(self, index):
        """
        Augment the data by adding negative binomial noise to the original expression values.
        """
        # Call the superclass's __getitem__ method to get the original data
        original_data = super().__getitem__(index)
        original_expression = (
            original_data["x"].numpy().flatten()
        )  # Convert to numpy array

        # Estimate mean and dispersion (overdispersion parameter) for the distribution
        mean_expression = original_expression
        dispersion = (
            1 / self.adata.varm["overdispersion"]
            if "overdispersion" in self.adata.varm
            else np.ones_like(mean_expression)
        )

        augmented_data_list = []
        for _ in range(self.n_augmentations):
            noise = np.random.negative_binomial(
                n=dispersion, p=mean_expression / (mean_expression + dispersion)
            )
            augmented_expression = original_expression + noise

            # Ensure no negative values (counts cannot be negative)
            augmented_expression = np.maximum(augmented_expression, 0)

            # Replace the "x" key in the original data with the augmented expression
            augmented_data = original_data.copy()
            augmented_data["x"] = torch.tensor(
                augmented_expression, dtype=torch.float32
            )

            augmented_data_list.append(augmented_data)

        return augmented_data_list

    def augment_on_the_fly(self, index):
        """Augment a single cell by sampling from the same cell type, performing a random walk on the kNN graph, or adding negative binomial noise."""
        if self.augmentation_type == "cell_type":
            return self.cell_type_augment(index)
        elif self.augmentation_type == "knn" or self.augmentation_type == "scanpy_knn":

            return self.knn_augment(index)

        elif self.augmentation_type == "community":

            return self.community_augment(index)
        elif self.augmentation_type == "nb":
            return self.negative_binomial_augment(index)
        elif self.augmentation_type == "mix":
            return (
                self.cell_type_augment(index)
                if random.random() < self.supervised_ratio
                else self.knn_augment(index)
            )
        else:
            raise ValueError(f"Invalid augmentation_type: {self.augmentation_type}")

    def __getitem__(self, index):
        if isinstance(index, int):
            # Single index
            augmented_data_list = self.augment_on_the_fly(index)
        elif isinstance(index, slice):
            # Slice of indices
            indices = range(*index.indices(len(self)))

            augmented_data_list = []
            for idx in indices:
                augmented_data_list.extend(self.augment_on_the_fly(idx))
        else:
            raise TypeError("Invalid index type")

        if self.augmentation_type == "nb":
            combined_data = self.combine_augmented_data(augmented_data_list)
        else:
            # Fetch the augmented data using the parent's __getitem__ method
            augmented_data_list = [
                super().__getitem__(aug_idx) for aug_idx in augmented_data_list
            ]
            combined_data = self.combine_augmented_data(augmented_data_list)

        return combined_data

    def combine_augmented_data(self, augmented_data_list):
        """Combine the list of augmented data into a single batch."""
        keys_to_stack = [
            "x",
            "labeled",
            "sizefactor",
            "batch",
            "combined_batch",
            # "study",
            "celltypes",
        ]

        combined_data = {}
        for key in keys_to_stack:
            if key in augmented_data_list[0]:
                combined_data[key] = torch.stack(
                    [data[key] for data in augmented_data_list]
                )

        return combined_data


def reshape_and_reorder_dict(data_dict):
    """
    Reshape and reorder the tensors in the dictionary.
    Handles tensors with different shapes by applying reshaping accordingly.
    """
    reshaped_dict = {}

    for key, tensor in data_dict.items():
        # Store the reshaped tensor in the dictionary
        reshaped_dict[key] = reshape_and_reorde_tensor(tensor)
    return reshaped_dict


def reshape_and_reorde_tensor(tensor):
    batch_size, num_augmentations = tensor.shape[:2]
    feature_dims = tensor.shape[2:]

    # Permute the tensor to bring augmentations to the first dimension
    permuted_tensor = tensor.permute(1, 0, *range(2, len(tensor.shape)))

    # Reshape to combine the augmentation and batch dimensions
    reshaped_tensor = permuted_tensor.reshape(
        num_augmentations * batch_size, *feature_dims
    )
    return reshaped_tensor
