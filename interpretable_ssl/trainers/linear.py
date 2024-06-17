import scanpy as sc
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
def train_linear_classifier(X, y, test_size=0.2, random_state=42):
    """
    Train a linear classifier for cell type classification.

    Parameters:
    adata (AnnData): The annotated data matrix of shape `n_obs` × `n_vars`.
    cell_type_col (str): The column name in `adata.obs` that contains cell type labels.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed.

    Returns:
    model (LogisticRegression): The trained logistic regression model.
    report (str): Classification report on the test set.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Train a linear classifier (logistic regression)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Generate a classification report
    report = classification_report(y_test, y_pred, target_names=np.unique(y),  output_dict=True)

    return pd.DataFrame(report)

