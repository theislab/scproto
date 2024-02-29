from pancras_data import *
import scanpy as sc

if __name__ == "__main__":
    # load data
    pancras = PancrasDataset()
    
    # define number of pc-components
    n_comps = 8
    
    # calculate pca 
    sc.tl.pca(pancras.adata, n_comps=n_comps)
    data_path = '.'.join(str(get_data_path()).split('.')[:-1]) + '-pca.h5ad'
    
    pancras.adata.write_h5ad(data_path)
    
    
    