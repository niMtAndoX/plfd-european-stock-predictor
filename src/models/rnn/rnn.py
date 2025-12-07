from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_model import BaseModel


# ---------------------------------------------------------
# Backbone: Basic RNN + FC head
# ---------------------------------------------------------

class RNNBackbone(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """
        x: (B, T, F)
        """
        out, h_n = self.rnn(x)      # out: (B, T, H)
        last = out[:, -1, :]        # last time step
        return self.fc(last)        # (B, 1)
        

# ---------------------------------------------------------
# RNNModel compatible with your BaseModel
# ---------------------------------------------------------

class RNNModel(BaseModel):
    def __init__(
        self,
        name: str = "RNN",
        seq_len: int = 30,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        device: str = None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None

    # -----------------------------------------------------
    # Sliding Windows (same as CNN/SACLSTM)
    # -----------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        df = df.dropna().copy()

        X_raw = df[self.feature_cols].values.astype(float)
        y_raw = df[self.target_col].values.astype(float)

        n = len(df)
        num_samples = n - self.seq_len

        X = []
        y = []

        for i in range(num_samples):
            X.append(X_raw[i : i + self.seq_len])
            y.append(y_raw[i + self.seq_len])

        X = np.array(X)                        # (B, T, F)
        y = np.array(y)                        # (B,)

        num_features = X.shape[2]

        if self.model is None:
            self.model = RNNBackbone(
                input_size=num_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)

        return X, y

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)

        loader = DataLoader(TensorDataset(X_train, y_train),
                            batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for Xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"[{self.name}] Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(loader):.6f}")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X).cpu().numpy().flatten()
        return preds

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        mse = float(np.mean((preds - y_test) ** 2))
        mae = float(np.mean(np.abs(preds - y_test)))
        rmse = mse ** 0.5
        return {"mse": mse, "mae": mae, "rmse": rmse}
