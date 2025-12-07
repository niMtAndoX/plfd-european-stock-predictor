
from __future__ import annotations
from typing import Dict, List

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from base_model import BaseModel


# ---------------------------------------------------------
# Segment-Correlation (single scale)
# ---------------------------------------------------------

class SegmentCorrelation(nn.Module):
    """
    Segment-wise attention:
    - Divide time axis into equal-length segments.
    - Compute correlations between segments (Qi, Kj).
    - Aggregate value segments with those weights.
    """

    def __init__(self, d_model: int, seg_len: int):
        super().__init__()
        self.d_model = d_model
        self.seg_len = seg_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.shape
        L = self.seg_len

        # ----- pad to multiple of L -----
        n_seg = math.ceil(T / L)
        pad_len = n_seg * L - T
        if pad_len > 0:
            pad = x[:, -1:, :].repeat(1, pad_len, 1)
            x_p = torch.cat([x, pad], dim=1)  # (B, T', D)
        else:
            x_p = x

        B, Tp, _ = x_p.shape

        # projections
        Q = self.q_proj(x_p)
        K = self.k_proj(x_p)
        V = self.v_proj(x_p)

        # reshape into segments
        Q = Q.view(B, n_seg, L, D)
        K = K.view(B, n_seg, L, D)
        V = V.view(B, n_seg, L, D)

        # flatten each segment
        Qf = Q.reshape(B, n_seg, L * D)  # (B, m, LD)
        Kf = K.reshape(B, n_seg, L * D)  # (B, n, LD)
        Vf = V.reshape(B, n_seg, L * D)  # (B, n, LD)

        # segment correlation scores: (B, m, n)
        scores = torch.bmm(Qf, Kf.transpose(1, 2)) / math.sqrt(L * D)
        weights = torch.softmax(scores, dim=-1)

        # aggregate values segment-wise
        out_flat = torch.bmm(weights, Vf)  # (B, m, LD)
        out_seg = out_flat.view(B, n_seg, L, D)
        out = out_seg.reshape(B, n_seg * L, D)  # (B, T', D)

        # remove padding
        if pad_len > 0:
            out = out[:, :T, :]

        out = self.out_proj(out)
        return out


# ---------------------------------------------------------
# Multi-Scale Segment-Correlation (MSSC)
# ---------------------------------------------------------

class MultiScaleSegmentCorrelation(nn.Module):
    """
    Fuse several SegmentCorrelation modules with different segment lengths.
    """

    def __init__(self, d_model: int, seg_lens: List[int]):
        super().__init__()
        self.scales = nn.ModuleList([
            SegmentCorrelation(d_model, L) for L in seg_lens
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [scale(x) for scale in self.scales]
        return sum(outs) / len(outs)


# ---------------------------------------------------------
# Preformer encoder layer (MSSC + FFN)
# ---------------------------------------------------------

class PreformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        seg_lens: List[int],
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mssc = MultiScaleSegmentCorrelation(d_model, seg_lens)
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
        # MSSC attention + residual
        attn_out = self.mssc(x)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward + residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


# ---------------------------------------------------------
# Preformer backbone: stacked encoder + prediction head
# ---------------------------------------------------------

class PreformerBackbone(nn.Module):
    def __init__(
        self,
        in_features: int,
        seq_len: int,
        out_len: int,
        d_model: int = 64,
        seg_lens: List[int] | None = None,
        num_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.out_len = out_len

        if seg_lens is None:
            seg_lens = [4, 8]  # example multi-scale segments

        self.input_proj = nn.Linear(in_features, d_model)

        self.layers = nn.ModuleList([
            PreformerEncoderLayer(
                d_model=d_model,
                seg_lens=seg_lens,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Use last time step representation to predict future horizon
        self.fc_out = nn.Linear(d_model, out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, F)
        returns: (B, out_len)
        """
        x = self.input_proj(x)  # (B, T, d_model)

        for layer in self.layers:
            x = layer(x)

        last = x[:, -1, :]      # (B, d_model)
        y_hat = self.fc_out(last)
        return y_hat


# ---------------------------------------------------------
# BaseModel wrapper
# ---------------------------------------------------------

class PreformerModel(BaseModel):
    def __init__(
        self,
        name: str = "Preformer",
        seq_len: int = 30,
        out_len: int = 1,
        d_model: int = 64,
        seg_lens: List[int] | None = None,
        num_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | None = None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.out_len = out_len
        self.d_model = d_model
        self.seg_lens = seg_lens or [4, 8]
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: nn.Module | None = None

    # -----------------------------------------------------
    # Data prep: seq_len history -> out_len future returns
    # -----------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        df_clean = df.dropna().copy()

        X_raw = df_clean[self.feature_cols].values.astype(float)
        y_raw = df_clean[self.target_col].values.astype(float)

        n = len(df_clean)
        num_samples = n - self.seq_len - self.out_len + 1
        if num_samples <= 0:
            raise ValueError(
                f"Not enough data for seq_len={self.seq_len} and out_len={self.out_len} (n={n})."
            )

        X_list, y_list = [], []
        for i in range(num_samples):
            X_list.append(X_raw[i : i + self.seq_len])
            y_list.append(y_raw[i + self.seq_len : i + self.seq_len + self.out_len])

        X = np.array(X_list)  # (B, T, F)
        y = np.array(y_list)  # (B, out_len)

        in_features = X.shape[2]
        if self.model is None:
            self.model = PreformerBackbone(
                in_features=in_features,
                seq_len=self.seq_len,
                out_len=self.out_len,
                d_model=self.d_model,
                seg_lens=self.seg_lens,
                num_layers=self.num_layers,
                d_ff=self.d_ff,
                dropout=self.dropout,
            ).to(self.device)

        return X, y

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

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
            print(f"[{self.name}] Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(loader):.6f}")

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------
    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X).cpu().numpy()  # (B, out_len)
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
