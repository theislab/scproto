import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scanpy as sc

def calculate_umap(embeddings, prototypes=None):
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
        combined_adata.obs['type'] = ['cell'] * num_cells + ['prototype'] * num_prototypes

        # Perform UMAP on the combined data
        sc.pp.neighbors(combined_adata, use_rep='X')
        sc.tl.umap(combined_adata)

        # Extract UMAP embeddings
        umap_embedding = combined_adata.obsm['X_umap']
        cell_umap = umap_embedding[:num_cells]
        prototype_umap = umap_embedding[num_cells:]
    else:
        # Perform UMAP on the embeddings only
        adata = sc.AnnData(embeddings)
        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.umap(adata)
        
        # Extract UMAP embeddings
        umap_embedding = adata.obsm['X_umap']
        cell_umap = umap_embedding
        prototype_umap = None
    
    return cell_umap, prototype_umap

def plot_umap(cell_umap, prototype_umap, cell_types, study_labels):
    # Plot UMAP embeddings colored by cell type
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot cell embeddings colored by cell type
    unique_cell_types = np.unique(cell_types)
    
    # Generate a list of unique colors for cell types
    unique_colors = plt.cm.get_cmap('tab20', len(unique_cell_types))
    colors = ListedColormap(unique_colors(np.linspace(0, 1, len(unique_cell_types))))
    
    for i, cell_type in enumerate(unique_cell_types):
        indices = np.where(cell_types == cell_type)[0]
        axes[0].scatter(cell_umap[indices, 0], cell_umap[indices, 1], label=cell_type, color=colors(i), alpha=0.6, s=20)
    
    if prototype_umap is not None:
        # Highlight prototypes
        axes[0].scatter(prototype_umap[:, 0], prototype_umap[:, 1], color='white', edgecolor='black', s=100, marker='o', label='Prototypes')
    
    axes[0].set_title('UMAP of Cell Embeddings with Cell Types Highlighted')
    axes[0].set_xlabel('UMAP Dimension 1')
    axes[0].set_ylabel('UMAP Dimension 2')
    
    # Plot cell embeddings colored by study
    unique_studies = np.unique(study_labels)
    
    # Generate a list of unique colors for studies
    unique_colors_studies = plt.cm.get_cmap('tab20', len(unique_studies))
    colors_studies = ListedColormap(unique_colors_studies(np.linspace(0, 1, len(unique_studies))))
    
    for i, study in enumerate(unique_studies):
        indices = np.where(study_labels == study)[0]
        axes[1].scatter(cell_umap[indices, 0], cell_umap[indices, 1], label=study, color=colors_studies(i), alpha=0.6, s=20)
    
    if prototype_umap is not None:
        # Highlight prototypes
        axes[1].scatter(prototype_umap[:, 0], prototype_umap[:, 1], color='white', edgecolor='black', s=100, marker='o', label='Prototypes')
    
    axes[1].set_title('UMAP of Cell Embeddings with Studies Highlighted')
    axes[1].set_xlabel('UMAP Dimension 1')
    axes[1].set_ylabel('UMAP Dimension 2')
    
    # Adjust layout to make space for the legends
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    
    # Adding legends below the plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.25, -0.2), ncol=3, title='Cell Types')
    
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.75, -0.2), ncol=3, title='Studies')
    
    plt.show()

# Example usage:
# cell_umap, prototype_umap = calculate_umap(embeddings, prototypes)
# plot_umap(cell_umap, prototype_umap, cell_types, study_labels)
