import logging
import random
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import get_worker_info
import pickle as pkl
import numpy as np


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphHandler:

    def __init__(self, original_incidies) -> None:
        self.graph = None
        adata_size = len(original_incidies)
        self.adata_size = adata_size
        self.max_knn_size = 30000

        self.set_original_ind(original_incidies)
        
    def set_original_ind(self, original_incidies):
        if self.generate_graph():
            self.original_incidies = list(range(self.adata_size))
        else:
            self.original_incidies = original_incidies

    def load_graph(self):
        logger.info("loading the graph")
        g = pkl.load(open("data/hlca_knn.pkl", "rb"))
        logger.info(f"done, loaded graph with {len(g[0])} nodes")
        g = self.refine_graph(g)
        self.have_complete_graph = True
        return g

    def refine_graph(self, g):
        """
        Filters indices and distances to retain only the nodes in nodes_to_keep.

        Parameters:
        - indices: 2D array of neighbor indices for each node
        - distances: 2D array of distances corresponding to each neighbor
        - nodes_to_keep: List of node indices to retain in the graph

        Returns:
        - filtered_indices: 2D array of filtered neighbor indices
        - filtered_distances: 2D array of filtered neighbor distances
        """
        indices, distances = g
        nodes_to_keep = self.original_incidies

        # # Step 1: Filter rows based on nodes_to_keep
        # indices_filtered = indices[nodes_to_keep]
        # distances_filtered = distances[nodes_to_keep]

        # Step 2: Filter each row of neighbors, keeping only those in nodes_to_keep
        mask = np.isin(indices, nodes_to_keep)  # Boolean mask for valid neighbors
        filtered_indices = [row[mask[i]] for i, row in enumerate(indices)]
        filtered_distances = [row[mask[i]] for i, row in enumerate(distances)]

        # Convert to numpy arrays and return
        return np.array(filtered_indices, dtype=object), np.array(
            filtered_distances, dtype=object
        )

    def map_graph_index(self, index):
        return self.original_incidies[index]

    def get_adata_index(self, knn_index):
        return self.original_incidies.index(knn_index)

    def generate_graph(self):
        return self.adata_size < self.max_knn_size
