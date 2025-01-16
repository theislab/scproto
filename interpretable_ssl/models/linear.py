import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm

class LinearClassifier:
    def __init__(self, embeddings=None, labels=None, batch_size=32, test_size=0.2, random_state=42, lr=0.001, epochs=20,
                 X_train=None, X_test=None, y_train=None, y_test=None):
        """
        Initialize the classifier with embeddings, labels, and hyperparameters.
        Args:
            embeddings (np.array or torch.Tensor): The input embeddings.
            labels (pd.Series or list): The corresponding labels for the embeddings.
            batch_size (int): Size of the batches for training and evaluation.
            test_size (float): Proportion of the data to use for testing.
            random_state (int): Seed for reproducibility.
            lr (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            X_train, X_test, y_train, y_test: Optional pre-split training and testing data.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.lr = lr
        self.epochs = epochs

        self.embeddings = torch.tensor(embeddings, dtype=torch.float32) if embeddings is not None else None
        self.labels = labels
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        """Prepare the data by encoding labels and creating datasets."""
        if self.X_train is not None and self.X_test is not None:
            # Assume y_train and y_test are provided with X_train and X_test
            self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
            self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
            self.y_train, self.y_test = self._encode_labels(self.y_train, self.y_test)
        else:
            # Split embeddings and labels
            numeric_labels = self._encode_labels(self.labels)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.embeddings, numeric_labels, test_size=self.test_size, random_state=self.random_state
            )

        self.train_data = CustomDataset(self.X_train, self.y_train)
        self.test_data = CustomDataset(self.X_test, self.y_test)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

    def _encode_labels(self, *label_sets):
        """Encode labels into numeric format."""
        le = LabelEncoder()
        labels = [torch.tensor(le.fit_transform(labels), dtype=torch.long) for labels in label_sets] if label_sets else le.fit_transform(self.labels)
        self.classes_ = le.classes_
        return labels

    def _build_model(self):
        """Build a simple linear classifier."""
        input_size = self.X_train.shape[1]
        output_size = len(self.classes_)
        self.model = nn.Linear(input_size, output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        """Train the linear classifier using tqdm for progress bar."""
        self.model.train()
        
        # Loop over the epochs
        for epoch in range(self.epochs):
            running_loss = 0.0
            # Initialize tqdm for the progress bar, set total number of batches
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.epochs}", unit='batch', leave=False) as pbar:
                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()  # Zero the parameter gradients
                    outputs = self.model(inputs)  # Forward pass
                    loss = self.criterion(outputs, labels)  # Loss calculation
                    loss.backward()  # Backpropagation
                    self.optimizer.step()  # Optimization

                    running_loss += loss.item()
                    # Update the tqdm progress bar with the running loss
                    pbar.set_postfix(loss=f"{running_loss/len(self.train_loader):.4f}")
                    pbar.update(1)  # Move the progress bar forward by one batch

            # print(f"Epoch {epoch+1}/{self.epochs} completed. Average Loss: {running_loss/len(self.train_loader):.4f}")

    def _evaluate_on_loader(self, loader):
        """Helper function to evaluate the model on a given DataLoader."""
        y_true, y_pred = [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        return y_true, y_pred

    def evaluate(self):
        """Evaluate the model and return F1 scores for each class and overall metrics as a DataFrame."""
        # Get predictions and true labels for train and test sets
        y_true_train, y_pred_train = self._evaluate_on_loader(self.train_loader)
        y_true_test, y_pred_test = self._evaluate_on_loader(self.test_loader)

        # Calculate F1 scores for each class and overall metrics
        f1_train_per_class = f1_score(y_true_train, y_pred_train, average=None)
        f1_test_per_class = f1_score(y_true_test, y_pred_test, average=None)

        # Calculate macro, micro, and weighted F1 scores
        f1_train_macro = f1_score(y_true_train, y_pred_train, average='macro')
        f1_train_micro = f1_score(y_true_train, y_pred_train, average='micro')
        f1_train_weighted = f1_score(y_true_train, y_pred_train, average='weighted')

        f1_test_macro = f1_score(y_true_test, y_pred_test, average='macro')
        f1_test_micro = f1_score(y_true_test, y_pred_test, average='micro')
        f1_test_weighted = f1_score(y_true_test, y_pred_test, average='weighted')

        # # Create DataFrame for F1 scores
        # f1_df = pd.DataFrame({
        #     'Class': self.classes_,
        #     'Train F1 Score': f1_train_per_class,
        #     'Test F1 Score': f1_test_per_class
        # })

        # Append overall metrics
        overall_metrics = pd.DataFrame({
            'Class': ['macro', 'micro', 'weighted'],
            # 'Train F1 Score': [f1_train_macro, f1_train_micro, f1_train_weighted],
            'F1 Score': [f1_test_macro, f1_test_micro, f1_test_weighted]
        })

        # f1_df = pd.concat([f1_df, overall_metrics], ignore_index=True)

        return overall_metrics

class CustomDataset(Dataset):
    def __init__(self, embeddings, labels):
        """
        Custom PyTorch Dataset for embeddings and labels.
        Args:
            embeddings (torch.Tensor): The input embeddings.
            labels (torch.Tensor or np.array): The corresponding labels.
        """
        self.embeddings = embeddings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]