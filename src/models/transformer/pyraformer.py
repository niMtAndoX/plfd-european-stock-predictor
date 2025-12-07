# pyraformer_model.py
#
# Simplified Pyraformer for time series forecasting.
# Compatible with your BaseModel interface.

from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from base_model import BaseModel


# ---------------------------------------------------------
# Positional Encoding
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)


# ---------------------------------------------------------
# Pyramid Attention Module (PAM, simplified)
# ---------------------------------------------------------

class PyramidAttention(nn.Module):
    """
    Simplified Pyramid Attention:
    - Queries: full-resolution sequence.
    - Keys/Values: downsampled in time via avg pooling (pyramid scale).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, pool_stride: int = 2):
        super().__init__()
        self.pool_stride = pool_stride
        self.pool = nn.AvgPool1d(kernel_size=pool_stride, stride=pool_stride)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # (B, T, D)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.shape

        # Downsample in time to build pyramid representation
        x_pool = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, T', D)

        # Queries at full scale, keys/values at pyramid scale
        q = x
        k = x_pool
        v = x_pool

        out, _ = self.attn(q, k, v)   # (B, T, D)
        return out


# ---------------------------------------------------------
# Pyraformer Block (CSCM + PAM + FFN, with Add & Norm)
# ---------------------------------------------------------

class PyraformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        pool_stride: int = 2,
    ):
        super().__init__()

        self.pam = PyramidAttention(d_model, n_heads, dropout, pool_stride)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PAM + residual
        pam_out = self.pam(x)
        x = self.norm1(x + self.dropout1(pam_out))

        # Feed-forward + residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x


# ---------------------------------------------------------
# Pyraformer Backbone
# ---------------------------------------------------------

class PyraformerBackbone(nn.Module):
    def __init__(
        self,
        obs_size: int,          # size of observation channel (usually 1)
        cov_size: int,          # number of covariate features
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
        pool_stride: int = 2,
        out_len: int = 1,
    ):
        super().__init__()

        self.out_len = out_len

        # Embeddings
        self.obs_embed = nn.Linear(obs_size, d_model)
        self.cov_embed = nn.Linear(cov_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Stack of Pyraformer blocks
        self.blocks = nn.ModuleList([
            PyraformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                pool_stride=pool_stride,
            )
            for _ in range(num_layers)
        ])

        # Prediction Strategy 1: Gather Features (use last-step representation)
        self.fc_out = nn.Linear(d_model, out_len)

    def forward(self, obs: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, T, 1)          -> observations (e.g. target series history)
        cov: (B, T, F_cov)      -> covariates (features)
        """
        # Embedding + sum (like in figure)
        h_obs = self.obs_embed(obs)       # (B, T, D)
        h_cov = self.cov_embed(cov)       # (B, T, D)
        x = h_obs + h_cov

        x = self.pos_encoding(x)

        for block in self.blocks:
            x = block(x)                  # (B, T, D)

        # Gather features: use last timestep representation
        last = x[:, -1, :]                # (B, D)
        y_hat = self.fc_out(last)         # (B, out_len)
        return y_hat


# ---------------------------------------------------------
# PyraformerModel wrapper (BaseModel)
# ---------------------------------------------------------

class PyraformerModel(BaseModel):
    def __init__(
        self,
        name: str = "Pyraformer",
        seq_len: int = 30,
        out_len: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
        pool_stride: int = 2,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | None = None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.out_len = out_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.pool_stride = pool_stride
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: nn.Module | None = None

    # -----------------------------------------------------
    # Data prep:
    #   - obs: past target values (Return_t)
    #   - cov: engineered features (feature_cols)
    # -----------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        df_clean = df.dropna().copy()

        cov_raw = df_clean[self.feature_cols].values.astype(float)
        obs_raw = df_clean[self.target_col].values.astype(float)

        n = len(df_clean)
        num_samples = n - self.seq_len - self.out_len + 1
        if num_samples <= 0:
            raise ValueError(
                f"Not enough data for seq_len={self.seq_len} and out_len={self.out_len} (n={n})."
            )

        X_obs, X_cov, y = [], [], []
        for i in range(num_samples):
            X_cov.append(cov_raw[i : i + self.seq_len])                 # (seq_len, F_cov)
            X_obs.append(obs_raw[i : i + self.seq_len].reshape(-1, 1))  # (seq_len, 1)
            y.append(obs_raw[i + self.seq_len : i + self.seq_len + self.out_len])

        X_obs = np.array(X_obs)   # (B, T, 1)
        X_cov = np.array(X_cov)   # (B, T, F_cov)
        y = np.array(y)           # (B, out_len)

        obs_size = 1
        cov_size = X_cov.shape[2]

        if self.model is None:
            self.model = PyraformerBackbone(
                obs_size=obs_size,
                cov_size=cov_size,
                d_model=self.d_model,
                n_heads=self.n_heads,
                num_layers=self.num_layers,
                d_ff=self.d_ff,
                dropout=self.dropout,
                pool_stride=self.pool_stride,
                out_len=self.out_len,
            ).to(self.device)

        return (X_obs, X_cov), y

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_obs_train, X_cov_train = X_train
        X_obs_train = torch.tensor(X_obs_train, dtype=torch.float32).to(self.device)
        X_cov_train = torch.tensor(X_cov_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_obs_train, X_cov_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for obs_b, cov_b, y_b in loader:
                optimizer.zero_grad()
                preds = self.model(obs_b, cov_b)
                loss = criterion(preds, y_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"[{self.name}] Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    def predict(self, X):
        X_obs, X_cov = X
        X_obs = torch.tensor(X_obs, dtype=torch.float32).to(self.device)
        X_cov = torch.tensor(X_cov, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_obs, X_cov).cpu().numpy()
        return preds

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        preds = self.predict(X_test)
        preds_flat = preds.reshape(-1)
        y_flat = y_test.reshape(-1)

        mse = float(np.mean((preds_flat - y_flat) ** 2))
        mae = float(np.mean(np.abs(preds_flat - y_flat)))
        rmse = float(mse ** 0.5)
        return {"mse": mse, "mae": mae, "rmse": rmse}
