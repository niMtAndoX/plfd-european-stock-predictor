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
        self.kernel_size = int(kernel_size)

    def forward(self, series: torch.Tensor):
        """
        series: (B, T, D)
        returns: seasonal, trend, both (B, T, D)
        """
        B, T, D = series.shape
        k = self.kernel_size

        # (B, D, T) for easier time-axis operations
        x = series.permute(0, 2, 1)  # (B, D, T)

        # pad both ends with edge values, length k//2 on each side
        half = k // 2
        left_pad = x[:, :, :1].expand(-1, -1, half)
        right_pad = x[:, :, -1:].expand(-1, -1, half)
        x_padded = torch.cat([left_pad, x, right_pad], dim=2)  # (B, D, T + 2*half)

        # cumulative sum along time -> moving average over windows of size k
        cumsum = torch.cumsum(x_padded, dim=2)
        # window sum: sum[t : t+k] = cumsum[t+k-1] - cumsum[t-1]
        # build indices so output has exactly length T
        start = 0
        end = start + k
        # cumsum has length T + 2*half; we want T windows
        # window_sums: (B, D, T)
        window_sums = cumsum[:, :, start + k - 1 : start + k - 1 + T] - torch.cat(
            [torch.zeros_like(cumsum[:, :, :1]), cumsum[:, :, :T - 1]], dim=2
        )

        trend = window_sums / float(k)          # (B, D, T)
        trend = trend.permute(0, 2, 1)          # (B, T, D)

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

    def _auto_corr(self, Q, K, V):
        """
        Q, K, V: (B, T, H, Hd) with the same T
        """
        B, T, H, Hd = K.shape

        # FFT along time
        Q_fft = torch.fft.rfft(Q, dim=1)      # (B, T_f, H, Hd)
        K_fft = torch.fft.rfft(K, dim=1)      # (B, T_f, H, Hd)

        AC = Q_fft * torch.conj(K_fft)
        corr = torch.fft.irfft(AC, n=T, dim=1)  # (B, T, H, Hd)

        out = corr * V                         # (B, T, H, Hd)
        B2, T2, H2, Hd2 = out.shape
        out = out.reshape(B2, T2, H2 * Hd2)    # (B, T, D)
        return self.out(out)

    def forward(self, x: torch.Tensor):
        """
        Self auto-correlation:
        x: (B, T, D) -> (B, T, D)
        """
        B, T, D = x.shape
        H = self.n_heads
        Hd = self.head_dim

        Q = self.query(x).view(B, T, H, Hd)
        K = self.key(x).view(B, T, H, Hd)
        V = self.value(x).view(B, T, H, Hd)

        return self._auto_corr(Q, K, V)

    def cross_attend(self, query: torch.Tensor, memory: torch.Tensor):
        """
        Cross auto-correlation:
        query:  (B, T_q, D)
        memory: (B, T_m, D)
        Returns: (B, T_q, D)
        """
        Bq, T_q, D = query.shape
        Bm, T_m, Dm = memory.shape
        assert Bq == Bm and D == Dm, "batch and feature dims must match"

        H = self.n_heads
        Hd = self.head_dim

        # If encoder length != decoder length, interpolate encoder along time
        if T_m != T_q:
            # (B, T_m, D) -> (B, D, T_m)
            mem = memory.permute(0, 2, 1)
            mem = torch.nn.functional.interpolate(
                mem, size=T_q, mode="linear", align_corners=False
            )  # (B, D, T_q)
            memory = mem.permute(0, 2, 1)  # (B, T_q, D)
            T_m = T_q

        # Now T_m == T_q == T
        Q = self.query(query).view(Bq, T_q, H, Hd)    # (B, T, H, Hd)
        K = self.key(memory).view(Bm, T_m, H, Hd)     # (B, T, H, Hd)
        V = self.value(memory).view(Bm, T_m, H, Hd)   # (B, T, H, Hd)

        return self._auto_corr(Q, K, V)


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
        # x: (B, T_dec, D), memory: (B, T_enc, D)
        seasonal, trend = self.decomp1(x)           # seasonal: (B, T_dec, D)
        attn_out = self.cross_attn.cross_attend(
            seasonal, memory
        )                                           # (B, T_dec, D)
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
        x = self.input_proj(x)  # (B, T_enc, d_model)

        # ------- Encoder -------
        memory = x                       # (B, T_enc, d_model)
        trend_sum = 0
        for layer in self.encoder_layers:
            seasonal, trend = layer(memory)   # both (B, T_enc, d_model)
            memory = seasonal
            trend_sum += trend[:, -1, :]      # accumulate last trend element

        # ------- Decoder -------
        B, T_enc, D = memory.shape
        T_dec = self.out_len

        # Decoder seasonal/trend inputs: zeros with length out_len
        seasonal_dec = torch.zeros(B, T_dec, D, device=memory.device, dtype=memory.dtype)
        trend_dec = torch.zeros_like(seasonal_dec)

        dec_x = seasonal_dec + trend_dec      # (B, T_dec, D)

        # Expand encoder trend summary across decoder horizon
        # trend_sum: (B, D) -> (B, T_dec, D)
        trend_base = trend_sum.unsqueeze(1).expand(B, T_dec, D)

        for layer in self.decoder_layers:
            seasonal, trend = layer(dec_x, memory)  # both (B, T_dec, D)
            dec_x = seasonal
            trend_base = trend_base + trend

        # Final output projection: combine seasonal + trend components
        y_full = self.projection(dec_x + trend_base).squeeze(-1)  # (B, T_dec)

        # Ensure the returned shape is (B, out_len):
        B, T_dec = y_full.shape
        if T_dec == self.out_len:
            y = y_full
        else:
            # Simple reduction: take the last out_len steps if T_dec > out_len,
            # or average over time and repeat to match out_len if T_dec < out_len.
            if T_dec > self.out_len:
                y = y_full[:, -self.out_len:]
            else:
                mean_step = y_full.mean(dim=1, keepdim=True)  # (B, 1)
                y = mean_step.repeat(1, self.out_len)         # (B, out_len)

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

        X = np.array(X)          # (B, seq_len, F)
        y = np.array(y)          # (B, out_len) or (B,)

        if y.ndim == 1:
            y = y.reshape(-1, self.out_len)

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
        # Lazily initialize backbone if prepare_data() was not called on this instance
        if self.model is None:
            X_arr = np.asarray(X_train)
            if X_arr.ndim != 3:
                raise ValueError(
                    f"{self.name}.fit expected X_train with shape (B, T, F), "
                    f"got {X_arr.shape}"
                )
            in_features = X_arr.shape[2]
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
            X_train = X_arr

        # Ensure y_train is (B, out_len) to match model output
        y_train = np.asarray(y_train)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, self.out_len)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(Xb)        # (B, out_len)
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
