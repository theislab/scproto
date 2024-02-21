import torch
from pathlib import Path
import requests
from autoencoder import *
import json


def calculate_mean_var_latent(model, data_loader):
    z_arr = []
    for batch, adata in enumerate(data_loader):
        z = model.encoder(adata.X)
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
            "query": genes,
        },
    )
    return r.json['result']

if __name__ == "__main__":

    device = get_device()

    # load data
    batch_size = 64
    data_path = get_data_path()

    print("loading data")
    data = sc.read_h5ad(data_path)
    input_dim = data[0].shape[1]
    data_loader = AnnLoader(data, batch_size=batch_size, use_cuda=device)

    # load model
    model_path = get_model_path()
    hidden_dim, latent_dim = 256, 128
    model = VariationalAutoencoder(input_dim, hidden_dim, latent_dim)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.to(device)

    # calculate all feature vectors
    # calculate mean and variance
    var, mean = calculate_mean_var_latent(model, data_loader)

    # decode the mean latent vector and find mean cell
    mean_cell = model.decoder(mean)

    # for each dimension of latent vector change it by 2 sigma of that dimension
    # decode these vectors and calculate difference with mean cell
    dim_cells = calculate_all_dimension_cells(mean, var, model)
    diff_mean_cell = dim_cells - mean_cell

    # find k mostly affected genes for each dim
    k = 3
    top_idx = torch.topk(diff_mean_cell, k).indices
    tops_genes = [list(data.var.index[idx.cpu()]) for idx in top_idx]

    # do downstream analysis on mostly affected genes (go analysis - pathway analysis)
    res = {}
    for genes in tops_genes:
        res[str(genes)] = downstream(genes)
        
    with open('downstream.json', 'w') as f:
        json.dump(res, f)