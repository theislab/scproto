from interpretable_ssl.pancras.dataset import *
import scanpy as sc

def main():
    # load data
    pancras = PancrasDataset()
    
    # define number of pc-components
    n_comps = 8
    
    # calculate pca 
    sc.tl.pca(pancras.adata, n_comps=n_comps)
    data_path = '.'.join(str(get_data_path()).split('.')[:-1]) + '-pca.h5ad'
    
    pancras.adata.write_h5ad(data_path)
    
    
    