# prepare data
# load saved model
# get data representation
# plot umap
import scanpy as sc

def plot_umap(adata, use_rep):
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['cell_type', 'study'])