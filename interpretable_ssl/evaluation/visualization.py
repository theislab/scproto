# prepare data
# load saved model
# get data representation
# plot umap
import scanpy as sc

import scanpy as sc
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def calculate_umap(adata, prototypes, latent_key='latent'):
    num_cells = adata.shape[0]
    num_prototypes = prototypes.shape[0]
    
    # Combine embeddings and prototypes for UMAP
    combined_data = np.vstack((adata.obsm[latent_key], prototypes))
    combined_adata = sc.AnnData(combined_data)
    combined_adata.obs['type'] = ['cell'] * num_cells + ['prototype'] * num_prototypes

    # Perform UMAP on the combined data
    sc.pp.neighbors(combined_adata, use_rep='X')
    sc.tl.umap(combined_adata)

    # Extract UMAP embeddings
    umap_embedding = combined_adata.obsm['X_umap']
    cell_umap = umap_embedding[:num_cells]
    prototype_umap = umap_embedding[num_cells:]
    
    return cell_umap, prototype_umap



def plot_umap(cell_umap, prototype_umap, cell_types, studies):
    # Create a colormap with 50 distinct colors
    unique_colors = plt.cm.get_cmap('tab20', len(set(cell_types)))
    cell_type_colors = ListedColormap(unique_colors(np.linspace(0, 1, 50)))
    
    # Plot settings for studies
    unique_studies = np.unique(studies)
    study_colors = ListedColormap(plt.cm.get_cmap('tab20', len(unique_studies))(np.linspace(0, 1, len(unique_studies))))
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    
    # UMAP plot colored by cell type
    for i, cell_type in enumerate(np.unique(cell_types)):
        indices = np.where(cell_types == cell_type)[0]
        axs[0].scatter(cell_umap[indices, 0], cell_umap[indices, 1], label=cell_type, color=cell_type_colors(i % 50), alpha=0.6, s=20)
    
    # Highlight prototypes in the cell type plot
    axs[0].scatter(prototype_umap[:, 0], prototype_umap[:, 1], color='white', edgecolor='black', s=100, marker='o', label='Prototypes')
    
    axs[0].set_title('UMAP by Cell Type')
    axs[0].set_xlabel('UMAP Dimension 1')
    axs[0].set_ylabel('UMAP Dimension 2')
    
    # UMAP plot colored by study
    for i, study in enumerate(unique_studies):
        indices = np.where(studies == study)[0]
        axs[1].scatter(cell_umap[indices, 0], cell_umap[indices, 1], label=study, color=study_colors(i), alpha=0.6, s=20)
    
    # Highlight prototypes in the study plot
    axs[1].scatter(prototype_umap[:, 0], prototype_umap[:, 1], color='white', edgecolor='black', s=100, marker='o', label='Prototypes')
    
    axs[1].set_title('UMAP by Study')
    axs[1].set_xlabel('UMAP Dimension 1')
    axs[1].set_ylabel('UMAP Dimension 2')
    
    # Adjust layout to accommodate legends below plots
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    # Create a legend for cell types
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.25, -0.15), fontsize=8, ncol=5)

    # Create a legend for studies
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.75, -0.15), fontsize=8, ncol=5)
    
    plt.show()

def plot_umap(adata, use_rep):
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['cell_type', 'study'])

