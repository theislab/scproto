import pandas as pd

# https://panglaodb.se/markers.html?cell_type=%27choose%27


def get_marker_genes():
    cell_marker_genes = {
        "CD4+ T cells": ["CD4", "STAT5A", "FOXP3", "IL2RA"],
        "CD8+ T cells": ["CD8A", "CD8B1", "GZMA", "PRF1"],
        "CD10+ B cells": ["MME", "CD19", "CD20", "CR2"],
        "CD14+ Monocytes": ["CD14", "LYZ", "CSF1R", "ITGAM"],
        "CD16+ Monocytes": ["FCGR3A", "FCGR3B", "NCR1", "LILRA1"],
        "CD20+ B cells": ["MS4A1", "CD19", "CR2", "PAX5"],
        "Erythrocytes": ["HBB", "HBA1", "HBA2", "EPOR"],
        "Erythroid progenitors": ["GATA1", "KLF1", "EPOR", "HBB"],
        "HSPCs": ["CD34", "PROM1", "THY1", "KIT"],
        "Megakaryocyte progenitors": ["ITGA2B", "GP9", "PF4", "MYB"],
        "Monocyte progenitors": ["CD14", "CSF1R", "ITGAM", "LYZ"],
        "Monocyte-derived dendritic cells": ["CD1C", "CD83", "CSF2", "IL3"],
        "NK cells": ["NCAM1", "KLRD1", "NKG7", "FCGR3A"],
        "NKT cells": ["NCAM1", "FCGR3A", "NKG7", "KLRB1"],
        "Plasma cells": ["SDC1", "CD38", "IGHG1", "XBP1"],
        "Plasmacytoid dendritic cells": ["CLEC4C", "IL3RA", "LILRA4", "IRF7"],
    }

    marker_genes = [cell_marker_genes[cell] for cell in cell_marker_genes]
    marker_genes = [
        marker_genes[i][j]
        for i in range(len(marker_genes))
        for j in range(len(marker_genes[i]))
    ]
    return cell_marker_genes, marker_genes


def get_marker_genes_idx(adata):

    _, marker_genes = get_marker_genes()
    all_gene_set = set(adata.var.index)
    adata_marker_genes = all_gene_set.intersection(marker_genes)
    marker_gene_idx = [list(all_gene_set).index(gene) for gene in adata_marker_genes]
    return marker_gene_idx, adata_marker_genes


def get_marker_genes_expression(cells, marker_gene_idx, marker_gene_names, index):
    expressions = cells[:, marker_gene_idx]
    df = pd.DataFrame(expressions, index=index)
    df.columns = marker_gene_names
    return df
