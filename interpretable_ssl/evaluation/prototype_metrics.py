import torch
import pandas as pd

class PrototypeAnalyzer:
    def __init__(self, emb, prototype_layer, adata):
        """
        Initialize with embeddings, prototype linear layer, and AnnData object.
        """
        self.emb = emb
        self.prototype_layer = prototype_layer
        self.adata = adata
        self.purity_df = None
        self.summary = {}

    def get_prototype_weights(self):
        """Extracts prototype weights from the PyTorch linear layer."""
        return self.prototype_layer.weight.data  # shape (300, 8)

    def assign_to_closest_prototype(self):
        """Assigns each embedding to the closest prototype based on Euclidean distance."""
        prototype_weights = self.get_prototype_weights()
        distances = torch.cdist(self.emb, prototype_weights)  # shape (n, 300)
        closest_prototypes = torch.argmin(distances, dim=1)  # shape (n,)
        return closest_prototypes

    def index_by_prototype(self, closest_prototypes):
        """Indexes embeddings by the prototype they are closest to."""
        num_prototypes = self.get_prototype_weights().shape[0]
        prototype_to_indices = {i: [] for i in range(num_prototypes)}
        for idx, prototype_id in enumerate(closest_prototypes):
            prototype_to_indices[prototype_id.item()].append(idx)
        return prototype_to_indices

    def calculate_purity(self, prototype_to_indices):
        """Calculates cell type purity for each prototype based on assigned embeddings in `adata`."""
        purity_results = {}
        for prototype_id, indices in prototype_to_indices.items():
            if indices:
                # Retrieve cell types for embeddings assigned to this prototype
                cell_types = self.adata.obs['cell_type'][indices]
                
                # Calculate the most frequent cell type and its count
                value_counts = cell_types.value_counts()
                most_common_count = value_counts.iloc[0]
                purity = most_common_count / len(indices)
                
                purity_results[prototype_id] = purity
            else:
                purity_results[prototype_id] = None  # No embeddings assigned to this prototype
        return purity_results

    def calculate_summary(self):
        """Calculates and returns summary statistics including purity and assignment counts."""
        # Assign embeddings to closest prototypes and index by prototype
        closest_prototypes = self.assign_to_closest_prototype()
        prototype_to_indices = self.index_by_prototype(closest_prototypes)
        
        # Calculate purity for each prototype
        purity_results = self.calculate_purity(prototype_to_indices)
        
        # Create DataFrames for purity and counts
        self.purity_df = pd.DataFrame.from_dict(purity_results, orient='index', columns=['Purity'])
        counts_df = pd.DataFrame.from_dict({k: len(v) for k, v in prototype_to_indices.items()}, orient='index', columns=['Count'])
        
        # Add counts to purity DataFrame
        self.purity_df['Count'] = counts_df['Count']
        
        # Calculate embedding assignment statistics
        min_count = counts_df['Count'].min()
        max_count = counts_df['Count'].max()
        avg_count = counts_df['Count'].mean()
        
        # Calculate purity statistics
        valid_purity = self.purity_df.dropna(subset=['Purity'])
        min_purity = valid_purity['Purity'].min()
        max_purity = valid_purity['Purity'].max()
        avg_purity = valid_purity['Purity'].mean()
        
        # Weighted average purity
        total_count = valid_purity['Count'].sum()
        weighted_avg_purity = (valid_purity['Purity'] * valid_purity['Count']).sum() / total_count if total_count > 0 else None
        
        # Store summary
        self.summary = {
            "Min Embeddings per Prototype": min_count,
            "Max Embeddings per Prototype": max_count,
            "Avg Embeddings per Prototype": avg_count,
            "Min Purity": min_purity,
            "Max Purity": max_purity,
            "Avg Purity": avg_purity,
            "Weighted Avg Purity": weighted_avg_purity
        }
        self.calculate_closest_prototype_distances()
        return self.summary

    def calculate_closest_prototype_distances(self):
        """Calculates the distance of each prototype to its nearest neighboring prototype."""
        prototype_weights = self.get_prototype_weights()  # shape (300, 8)
        distances = torch.cdist(prototype_weights, prototype_weights)  # shape (300, 300)
        
        # Set diagonal to a large value to ignore self-distances
        distances.fill_diagonal_(float('inf'))
        
        # Find the minimum distance for each prototype to any other prototype
        closest_distances = distances.min(dim=1).values  # shape (300,)
        
        # Calculate min and average distance across all prototypes
        min_closest_distance = closest_distances.min().item()
        avg_closest_distance = closest_distances.mean().item()
        
        # Store results in the summary
        self.summary["Min Closest Prototype Distance"] = min_closest_distance
        self.summary["Avg Closest Prototype Distance"] = avg_closest_distance
        self.summary["max Closest Prototype Distance"] = closest_distances.max().item()

        return min_closest_distance, avg_closest_distance
        

