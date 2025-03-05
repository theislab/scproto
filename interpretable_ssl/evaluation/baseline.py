import scanpy as sc
from scib_metrics.benchmark import Benchmarker
import scvi
import pandas as pd
import os
import sys
sys.path.append("/content/drive/MyDrive/codes/Islander/src")
from scGraph import *

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def calculate_baseline_metrics(adata, name, test_studies, ref_epochs=200, query_epochs=5, n_latent=8, keys=["X_pca", "X_scvi"]):

    if 'X_pca' in keys:
        # claculate pca
        sc.pp.pca(adata, n_comps=50)

    # split to ref and query
    query_idx = adata.obs.study.isin(test_studies)
    ref = adata[~query_idx].copy()
    query = adata[query_idx].copy()

    if 'X_scvi' in keys:
        # calculate scvi
        # Step 2: Train scVI model and compute embeddings
        scvi.model.SCVI.setup_anndata(ref, batch_key="study")  # Change "batch" to your batch column
        model = scvi.model.SCVI(ref, n_latent=n_latent)
        model.train(ref_epochs)

        model_dir = f'{name}-scvi'
        # Step 2: Save the Trained Model
        os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists
        model.save(model_dir, overwrite=True)

        scvi.model.SCVI.prepare_query_anndata(query, model)
        scvi_query = scvi.model.SCVI.load_query_data(
            query,
            model,
        )
        scvi_query.train(max_epochs=query_epochs, plan_kwargs={"weight_decay": 0.0})
        # Get scVI latent representation
        adata.obsm["X_scvi"] = scvi_query.get_latent_representation(adata)

    # scib metrics
    bm = Benchmarker(
    adata,  # Your AnnData object
    batch_key="study",  # Replace with the correct batch annotation column
    label_key="cell_type",  # Replace with the correct cell type annotation column
    embedding_obsm_keys=keys   # Use PCA embedding for metrics
    )

    # Run the benchmark to compute scib-metrics
    bm.benchmark()

    # Get the results
    scib_results = bm.get_results(min_max_scale=False)
    scib_results = scib_results.iloc[:-1]

    # scgraph metrics
    adata_tmp_path = os.path.join("adata_tmp.h5ad")
    adata.write(adata_tmp_path)

    scgraph = scGraph(
        adata_path=adata_tmp_path,
        batch_key="study",
        label_key="cell_type",
        hvg=False,
        trim_rate=0.05,
        thres_batch=100,
        thres_celltype=10,
    )
    scgraph_results = scgraph.main(_obsm_list=keys)

    # classification
    f1 = evaluate_multiple_embeddings(ref, query, keys).T

    # save all
    return pd.concat([scib_results, f1, scgraph_results], axis=1)

def evaluate_multiple_embeddings(ref, query, keys, label_key="cell_type"):
    cd_cells = ['CD4+ T cells', 'CD8+ T cells']
    nk = ['NK cells', 'CD8+ T cells']
    all_cells = ref.obs[label_key].unique()

    results = {}

    for key in keys:
        print(f"Evaluating key: {key}")

        f1_cd = train_and_evaluate_classifier(ref, query, cd_cells, key=key, label_key=label_key)
        f1_nk = train_and_evaluate_classifier(ref, query, nk, key=key, label_key=label_key)
        f1_all = train_and_evaluate_classifier(ref, query, all_cells, key=key, label_key=label_key)

        results[key] = {"cd_cells": f1_cd, "nk": f1_nk, "all": f1_all}

    return pd.DataFrame(results)

def train_and_evaluate_classifier(adata_ref, adata_query, cell_types, key="X_pca", label_key="cell_type"):
    adata_ref_filtered = adata_ref[adata_ref.obs[label_key].isin(cell_types)].copy()
    adata_query_filtered = adata_query[adata_query.obs[label_key].isin(cell_types)].copy()
    if (len(adata_ref_filtered) == 0) or (len(adata_query_filtered) == 0):
        return None
    X_train = adata_ref_filtered.obsm[key]
    y_train = adata_ref_filtered.obs[label_key].values

    X_test = adata_query_filtered.obsm[key]
    y_test = adata_query_filtered.obs[label_key].values

    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    label_dict = {label: i for i, label in enumerate(unique_labels)}

    y_train_encoded = np.array([label_dict[label] for label in y_train])
    y_test_encoded = np.array([label_dict[label] for label in y_test])

    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train_encoded)

    y_pred = classifier.predict(X_test)
    f1 = f1_score(y_test_encoded, y_pred, average="macro")

    print(f"F1 Score: {f1:.4f}")
    
    return f1
