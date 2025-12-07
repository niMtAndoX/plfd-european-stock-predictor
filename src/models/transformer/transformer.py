from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from base_model import BaseModel


# ---------------------------------------------------------
# Positional Encoding (standard sine/cosine)
# ---------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, B, d_model)
        """
        T = x.size(0)
        x = x + self.pe[:T]
        return self.dropout(x)


# ---------------------------------------------------------
# Transformer backbone
# ---------------------------------------------------------

class TransformerBackbone(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # we'll use (T, B, E)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        """
        # project features
        x = self.input_proj(x)       # (B, T, d_model)

        # Transformer expects (T, B, d_model)
        x = x.permute(1, 0, 2)       # (T, B, d_model)

        x = self.pos_encoding(x)
        enc_out = self.encoder(x)    # (T, B, d_model)

        # Use last time step representation
        last = enc_out[-1, :, :]     # (B, d_model)

        y_hat = self.fc(last)        # (B, 1)
        return y_hat


# ---------------------------------------------------------
# TransformerModel wrapper (BaseModel)
# ---------------------------------------------------------

class TransformerModel(BaseModel):
    def __init__(
        self,
        name: str = "Transformer",
        seq_len: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | None = None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: nn.Module | None = None

    # -----------------------------------------------------
    # Data: sliding window, 1-step ahead
    # -----------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        df_clean = df.dropna().copy()

        X_raw = df_clean[self.feature_cols].values.astype(float)
        y_raw = df_clean[self.target_col].values.astype(float)

        n = len(df_clean)
        num_samples = n - self.seq_len
        if num_samples <= 0:
            raise ValueError(
                f"Not enough data for seq_len={self.seq_len} (n={n})."
            )

        X_list, y_list = [], []
        for i in range(num_samples):
            X_list.append(X_raw[i : i + self.seq_len])
            y_list.append(y_raw[i + self.seq_len])

        X = np.array(X_list)  # (B, T, F)
        y = np.array(y_list)  # (B,)

        num_features = X.shape[2]
        if self.model is None:
            self.model = TransformerBackbone(
                in_features=num_features,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
            ).to(self.device)

        return X, y

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(Xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"[{self.name}] Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")

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
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        preds = self.predict(X_test)
        mse = float(np.mean((preds - y_test) ** 2))
        mae = float(np.mean(np.abs(preds - y_test)))
        rmse = float(mse ** 0.5)
        return {"mse": mse, "mae": mae, "rmse": rmse}
