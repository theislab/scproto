import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report
from collections import Counter

def encode_labels(labels):
    """
    Encodes string labels into numeric values if necessary.
    """
    if isinstance(labels[0], str):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        return labels, label_encoder
    return labels, None

def drop_single_sample_labels(embeddings, labels):
    """
    Drops samples with labels that appear only once in the dataset.
    """
    # Count the occurrences of each label
    label_counts = Counter(labels)
    
    # Get the valid indices where the label appears more than once
    valid_indices = [i for i, label in enumerate(labels) if label_counts[label] > 1]
    
    # Filter embeddings and labels based on valid indices
    embeddings_filtered = embeddings[valid_indices]
    labels_filtered = labels[valid_indices]
    
    return embeddings_filtered, labels_filtered

def prepare_data(embeddings, labels, test_size=0.2):
    """
    Converts embeddings and labels to numpy if they are torch tensors.
    Splits the data into training and test sets.
    """
    # Move embeddings and labels to CPU if they are on CUDA and convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    embeddings, labels = drop_single_sample_labels(embeddings, labels)

    labels, label_encoder = encode_labels(labels)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=test_size, random_state=42, stratify=labels)
    
    return X_train, X_test, y_train, y_test, label_encoder

def train_knn_classifier(X_train, y_train, n_neighbors=5):
    """
    Trains a KNN classifier on the provided training data.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(knn, X_test, y_test, label_encoder=None):
    """
    Evaluates the trained model and returns a DataFrame with F1 scores for each class,
    as well as the micro, macro, and weighted F1 scores.
    """
    # Predict the test data
    y_pred = knn.predict(X_test)
    
    # Calculate F1 scores for each class and for micro, macro, and weighted averages
    class_f1_scores = f1_score(y_test, y_pred, average=None)
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Prepare a DataFrame for output
    results = {
        'Class': label_encoder.classes_ if label_encoder else np.unique(y_test),
        'F1 Score': class_f1_scores
    }
    results_df = pd.DataFrame(results)

    # Append micro, macro, and weighted F1 scores to the DataFrame
    summary_scores = pd.DataFrame({
        'Class': ['micro', 'macro', 'weighted'],
        'F1 Score': [micro_f1, macro_f1, weighted_f1]
    })
    results_df = pd.concat([results_df, summary_scores], ignore_index=True)
    
    return results_df

def knn_classifier_with_f1_report(embeddings, labels, test_size=0.2, n_neighbors=5):
    """
    Main function to handle data preparation, model training, and evaluation.
    Returns a DataFrame with the F1 scores for each class and overall scores.
    """
    # Prepare data and split it into train/test sets
    X_train, X_test, y_train, y_test, label_encoder = prepare_data(embeddings, labels, test_size)
    
    # Train the KNN classifier
    knn = train_knn_classifier(X_train, y_train, n_neighbors)
    
    # Evaluate the model and return results in a DataFrame
    f1_scores_df = evaluate_model(knn, X_test, y_test, label_encoder)
    
    return f1_scores_df

# Example usage:
# embeddings: torch.Tensor on CUDA
# labels: torch.Tensor on CUDA or strings in a pandas column
# f1_scores_df = knn_classifier_with_f1_report(embeddings, labels)
