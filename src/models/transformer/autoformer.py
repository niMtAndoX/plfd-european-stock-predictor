# autoformer_model.py
#
# Simplified Autoformer implementation (Auto-Correlation + Series Decomposition)
# Compatible with your BaseModel interface.

from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.base_model import BaseModel


# ---------------------------------------------------------
# Series decomposition: seasonal + trend
# ---------------------------------------------------------

class SeriesDecomp(nn.Module):
    """Moving average decomposition used in Autoformer."""
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )

    def forward(self, series: torch.Tensor):
        """
        series: (B, T, D)
        returns: seasonal, trend
        """
        trend = self.avg(series.permute(0,2,1)).permute(0,2,1)
        seasonal = series - trend
        return seasonal, trend


# ---------------------------------------------------------
# Auto-Correlation Attention (simplified)
# ---------------------------------------------------------

class AutoCorrelation(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        H = self.n_heads
        Hd = self.head_dim

        Q = self.query(x).view(B, T, H, Hd)
        K = self.key(x).view(B, T, H, Hd)
        V = self.value(x).view(B, T, H, Hd)

        # Auto-correlation uses frequency domain correlations
        Q_fft = torch.fft.rfft(Q, dim=1)
        K_fft = torch.fft.rfft(K, dim=1)
        
        # elementwise product in frequency domain
        AC = Q_fft * torch.conj(K_fft)
        
        # back to time domain
        corr = torch.fft.irfft(AC, n=T, dim=1)  # (B, T, H, Hd)

        # weight values by correlation
        out = corr * V
        out = out.reshape(B, T, D)
        return self.out(out)


# ---------------------------------------------------------
# Autoformer Encoder Layer
# ---------------------------------------------------------

class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, moving_avg: int):
        super().__init__()
        self.decomp1 = SeriesDecomp(moving_avg)
        self.attn = AutoCorrelation(d_model, n_heads)
        self.decomp2 = SeriesDecomp(moving_avg)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        seasonal, trend = self.decomp1(x)
        attn_out = self.attn(seasonal)
        seasonal2, _ = self.decomp2(seasonal + attn_out)
        ff_out = self.ff(seasonal2)
        return seasonal2 + ff_out, trend


# ---------------------------------------------------------
# Autoformer Decoder Layer
# ---------------------------------------------------------

class AutoformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, moving_avg: int):
        super().__init__()
        self.decomp1 = SeriesDecomp(moving_avg)
        self.cross_attn = AutoCorrelation(d_model, n_heads)
        self.decomp2 = SeriesDecomp(moving_avg)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, memory):
        seasonal, trend = self.decomp1(x)
        attn_out = self.cross_attn(memory)
        seasonal2, _ = self.decomp2(seasonal + attn_out)
        ff_out = self.ff(seasonal2)
        return seasonal2 + ff_out, trend


# ---------------------------------------------------------
# Autoformer Backbone
# ---------------------------------------------------------

class AutoformerBackbone(nn.Module):
    def __init__(
        self,
        in_features: int,
        seq_len: int,
        out_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 128,
        moving_avg: int = 25
    ):
        super().__init__()

        self.seq_len = seq_len
        self.out_len = out_len

        self.input_proj = nn.Linear(in_features, d_model)

        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, moving_avg)
            for _ in range(e_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            AutoformerDecoderLayer(d_model, n_heads, d_ff, moving_avg)
            for _ in range(d_layers)
        ])

        self.projection = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (B, seq_len, F)
        Returns predictions: (B, out_len)
        """

        x = self.input_proj(x)

        # ------- Encoder -------
        memory = x
        trend_sum = 0
        for layer in self.encoder_layers:
            seasonal, trend = layer(memory)
            memory = seasonal
            trend_sum += trend[:, -1, :]  # last trend element

        # ------- Decoder -------
        # Decoder input is zeros for seasonal and trend init
        seasonal_dec = torch.zeros_like(memory[:, -self.out_len:, :])
        trend_dec = torch.zeros_like(memory[:, -self.out_len:, :])

        dec_x = seasonal_dec + trend_dec

        for layer in self.decoder_layers:
            seasonal, trend = layer(dec_x, memory)
            dec_x = seasonal
            trend_sum = trend_sum.unsqueeze(1) + trend  # broadcast

        # Final output projection: combine seasonal + trend components
        y = self.projection(dec_x + trend_sum).squeeze(-1)  # (B, out_len)

        return y


# ---------------------------------------------------------
# BaseModel wrapper
# ---------------------------------------------------------

class AutoformerModel(BaseModel):
    def __init__(
        self,
        name="Autoformer",
        seq_len=30,
        out_len=1,
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        moving_avg=25,
        lr=1e-3,
        batch_size=32,
        epochs=20,
        device=None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.out_len = out_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None

    # -----------------------------------------------------
    # Prepare sliding-window data
    # -----------------------------------------------------
    def prepare_data(self, df):
        df = df.dropna().copy()

        X_raw = df[self.feature_cols].values.astype(float)
        y_raw = df[self.target_col].values.astype(float)

        n = len(df)
        num_samples = n - self.seq_len - self.out_len + 1

        X, y = [], []
        for i in range(num_samples):
            X.append(X_raw[i:i+self.seq_len])
            y.append(y_raw[i+self.seq_len:i+self.seq_len+self.out_len])

        X = np.array(X)
        y = np.array(y)

        in_features = X.shape[2]

        if self.model is None:
            self.model = AutoformerBackbone(
                in_features=in_features,
                seq_len=self.seq_len,
                out_len=self.out_len,
                d_model=self.d_model,
                n_heads=self.n_heads,
                e_layers=self.e_layers,
                d_layers=self.d_layers,
                d_ff=self.d_ff,
                moving_avg=self.moving_avg,
            ).to(self.device)

        return X, y

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        loader = DataLoader(TensorDataset(X_train, y_train),
                            batch_size=self.batch_size,
                            shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total = 0
            for Xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"[{self.name}] Epoch {epoch+1}/{self.epochs} Loss={total/len(loader):.6f}")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(X).cpu().numpy()

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        preds_flat = preds.reshape(-1)
        y_flat = y_test.reshape(-1)
        mse = float(np.mean((preds_flat - y_flat)**2))
        mae = float(np.mean(np.abs(preds_flat - y_flat)))
        rmse = mse**0.5
        return {"mse": mse, "mae": mae, "rmse": rmse}
