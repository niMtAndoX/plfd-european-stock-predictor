from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.base_model import BaseModel


class CNNModel(BaseModel):

    def __init__(
        self,
        name: str = "CNN",
        seq_len: int = 30,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        device: str = None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Will be created after we know num_features in prepare_data()
        self.model = None


    # ------------------------------------------------------------------
    # Build the CNN after we know num_features
    # ------------------------------------------------------------------
    def _build_network(self, num_features: int):
        self.model = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Conv1d(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Flatten(),
            nn.Linear((self.seq_len - 2*(self.kernel_size-1)) * self.hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)


    # ------------------------------------------------------------------
    # Prepare sliding window sequences
    # ------------------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        df_clean = df.dropna().copy()

        # Get numpy arrays
        X_raw = df_clean[self.feature_cols].values.astype(float)
        y_raw = df_clean[self.target_col].values.astype(float)

        num_samples = len(df_clean) - self.seq_len

        X = []
        y = []

        for i in range(num_samples):
            X.append(X_raw[i : i + self.seq_len])
            y.append(y_raw[i + self.seq_len])   # predict next day

        X = np.array(X)
        y = np.array(y)

        # Torch expects (batch, channels, sequence_len)
        # We currently have (batch, sequence_len, features)
        X = np.transpose(X, (0, 2, 1))

        # Build network dynamically
        num_features = X.shape[1]
        if self.model is None:
            self._build_network(num_features)

        return X, y


    # ------------------------------------------------------------------
    # Fit model
    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # If model not built (e.g. fit called without prepare_data on this instance),
        # build it from X_train shape.
        if self.model is None:
            # X_train is (B, C, T) here
            num_features = X_train.shape[1]
            self._build_network(num_features)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for Xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(Xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(loader):.6f}")


    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X).cpu().numpy().flatten()
        return preds


    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        mse = np.mean((preds - y_test) ** 2)
        mae = np.mean(np.abs(preds - y_test))
        rmse = mse ** 0.5
        return {"mse": mse, "mae": mae, "rmse": rmse}

