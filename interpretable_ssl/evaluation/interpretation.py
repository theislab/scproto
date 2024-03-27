import torch
import requests
from interpretable_ssl.models.autoencoder import *
from interpretable_ssl.pancras.dataset import PancrasDataset
from torch.utils.data import DataLoader
from interpretable_ssl.models.prototype_classifier import ProtClassifier
import pandas as pd
import interpretable_ssl.utils as utils

def calculate_mean_var_latent(model, data_loader):
    z_arr = []
    for x, _ in data_loader:
        z = model.encoder(x)
        z_arr.append(z)
    z_arr_tensor = torch.cat(z_arr, 0)
    var, mean = torch.var_mean(z_arr_tensor, dim=0)
    return var, mean


def calculate_all_dimension_cells(mean, var, model):
    dim_cnt = mean.shape[0]
    changed_latents = []
    sigma = torch.sqrt(var)
    for dim in range(dim_cnt):
        changed = mean.clone()
        changed[dim] += 2 * sigma[dim]
        changed_latents.append(changed)
    changed_latents_tensor = torch.stack(changed_latents, 0)
    dim_cells = model.decoder(changed_latents_tensor)
    return dim_cells


def downstream(gene_names):
    sources = ["GO:MF", "GO:CC", "GO:BP", "KEGG"]
    r = requests.post(
        url="https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        json={
            "organism": "hsapiens",
            "sources": sources,
            "query": gene_names,
        },
    )
    return r.json()['result']

def calculate_cell_mean_latent(cell, adata, model, device):
    cell_group = adata[adata.obs['cell_type'] == cell]
    cell_dataset = PancrasDataset(device, adata=cell_group)
    cell_loader = DataLoader(cell_dataset, 16)

    _, mean = calculate_mean_var_latent(model, cell_loader)
    return mean

def main():

    device = utils.get_device()

    # load data
    batch_size = 64
    print("loading data")
    pancras = PancrasDataset(device)
    data_loader = DataLoader(pancras, batch_size=batch_size, shuffle=False)

    # load model
    # define model
    num_prototypes, num_classes = 8, 14
    input_dim, hidden_dim, latent_dims = 4000, 64, 8
    model = ProtClassifier(
        num_prototypes=num_prototypes,
        num_classes=num_classes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dims=latent_dims,
    )
    model_path = vae_prototype_classifier.get_model_path(num_prototypes)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.to(device)

    
    # find all cell types
    cell_types = list(pancras.adata.obs['cell_type'].cat.categories)
    adata = pancras.adata
    res_df = []
    prot_cells = model.decoder(model.prototype_vectors)

    for cell in cell_types:
        
        # calculate all feature vectors
        # calculate mean and variance
        # var, mean = calculate_mean_var_latent(model, data_loader)
        mean = calculate_cell_mean_latent(cell, adata, model, device)

        # decode the mean latent vector and find mean cell
        mean_cell = model.decoder(mean)

        # # for each dimension of latent vector change it by 2 sigma of that dimension
        # # decode these vectors and calculate difference with mean cell
        # dim_cells = calculate_all_dimension_cells(mean, var, model)
        # diff_mean_cell = dim_cells - mean_cell
        
        # decode all prototypes and compare them with mean cell
        prot_diff = prot_cells - mean_cell

        # find k mostly affected genes for each dim
        k = 5
        top_idx = torch.topk(prot_diff, k).indices
        tops_genes = [list(pancras.adata.var.index[idx.cpu()]) for idx in top_idx]

        # do downstream analysis on mostly affected genes (go analysis - pathway analysis)
        res = []
        res_cnt = []
        for genes in tops_genes:
            genes_downstream = downstream(genes)
            res.append(genes_downstream)
            res_cnt.append(len(genes_downstream))
        res_df.append({'cell type':cell, 'gp': res, 'gp-cnt': res_cnt, 'top genes': tops_genes})
            
        # df = pd.DataFrame({'genes': tops_genes, 'biological': res, 'cnt': res_cnt})
        # df.to_csv('results/prototypes-interpretation.csv')
    pd.DataFrame(res_df).to_csv('prototypes.csv')
