import pandas as pd

def get_marker_genes_idx(adata):
    
    cell_marker_genes = {
        "CD4+ T cells": ["CD4", "IL2RA", "CCR5", "TBX21"],
        "CD8+ T cells": ["CD8A", "CD8B", "GZMA", "GZMB"],
        "CD10+ B cells": ["CD10", "CD19", "CD20", "MS4A1"],
        "CD14+ Monocytes": ["CD14", "CD68", "CD163"],
        "CD16+ Monocytes": ["CD16", "FCGR3A", "LYZ", "S100A9"],
        "CD20+ B cells": ["CD19", "CD20", "MS4A1", "CD79A"],
        "Erythrocytes": ["HBB", "ALAS2", "EPB42", "SLC4A1"],
        "Erythroid progenitors": ["GATA1", "KLF1", "EPOR", "ALAS2"],
        "HSPCs": ["CD34", "CD133", "CD90", "CD45"],
        "Megakaryocyte progenitors": ["CD41", "CD61", "MPL", "ITGA2B"],
        "Monocyte progenitors": ["CD34", "CD45RA", "CD117", "CD38"],
        "Monocyte-derived dendritic cells": ["CD1c", "CD11c", "CD86", "CD209"],
        "NK cells": ["CD56", "CD16", "CD94", "NKG2D"],
        "NKT cells": ["CD3", "CD56", "CD161", "TCR Vα24-Jα18"],
        "Plasma cells": ["CD138", "CD38", "CD19", "IRF4"],
        "Plasmacytoid dendritic cells": ["CD303", "CD304", "CD123", "TLR7"],
    }
    marker_genes = [cell_marker_genes[cell] for cell in cell_marker_genes]
    marker_genes = [marker_genes[i][j] for i in range(len(marker_genes)) for j in range(len(marker_genes[i]))]
    
    all_gene_set = set(adata.var.index)
    adata_marker_genes = all_gene_set.intersection(marker_genes)
    marker_gene_idx = [list(all_gene_set).index(gene) for gene in adata_marker_genes]
    return marker_gene_idx, adata_marker_genes

def get_marker_genes_expression(cells, marker_gene_idx, marker_gene_names, index):
    expressions = cells[:, marker_gene_idx]
    df = pd.DataFrame(expressions, index=index)
    df.columns = marker_gene_names
    return df