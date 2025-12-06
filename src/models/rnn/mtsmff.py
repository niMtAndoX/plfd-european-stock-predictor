from __future__ import annotations
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from base_model import BaseModel


# ---------------------------------------------------------
# Dot-product attention over encoder hidden states
# ---------------------------------------------------------

class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, enc_outputs: torch.Tensor):
        """
        query:      (B, H)
        enc_outputs:(B, T, H)
        returns:
            context: (B, H)
            attn_w:  (B, T)
        """
        # (B, T, H) x (B, H, 1) -> (B, T, 1)
        scores = torch.bmm(enc_outputs, query.unsqueeze(2)).squeeze(2)  # (B, T)
        attn_w = torch.softmax(scores, dim=1)                           # (B, T)
        context = torch.bmm(attn_w.unsqueeze(1), enc_outputs).squeeze(1)  # (B, H)
        return context, attn_w


# ---------------------------------------------------------
# Encoder–Decoder with attention (backbone)
# ---------------------------------------------------------

class MTSMFFBackbone(nn.Module):
    def __init__(
        self,
        in_features: int,
        enc_hidden: int = 64,
        dec_hidden: int | None = None,
        num_layers: int = 1,
        out_len: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden or enc_hidden
        self.num_layers = num_layers
        self.out_len = out_len

        self.encoder = nn.LSTM(
            input_size=in_features,
            hidden_size=self.enc_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attn = DotProductAttention()

        # Decoder as a single-layer LSTMCell
        self.decoder_cell = nn.LSTMCell(
            input_size=self.enc_hidden,  # we feed context vector
            hidden_size=self.dec_hidden,
        )

        self.fc_out = nn.Linear(self.dec_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_in, F)
        returns:
            y_hat: (B, out_len)
        """
        B = x.size(0)

        # ----- Encoder -----
        enc_outputs, (h_n, c_n) = self.encoder(x)     # enc_outputs: (B, T_in, H)
        # Take last layer hidden state as initial decoder state
        h_dec = h_n[-1]                               # (B, H)
        c_dec = c_n[-1]                               # (B, H)

        outputs = []

        for step in range(self.out_len):
            # ----- Attention over encoder outputs -----
            context, _ = self.attn(h_dec, enc_outputs)  # (B, H)

            # ----- One decoder step -----
            h_dec, c_dec = self.decoder_cell(context, (h_dec, c_dec))

            # Predict one step ahead
            y_step = self.fc_out(h_dec)  # (B, 1)
            outputs.append(y_step)

        # (out_len, B, 1) -> (B, out_len)
        y_hat = torch.stack(outputs, dim=1).squeeze(-1)
        return y_hat


# ---------------------------------------------------------
# MTSMFFModel: BaseModel wrapper
# ---------------------------------------------------------

class MTSMFFModel(BaseModel):
    def __init__(
        self,
        name: str = "MTSMFF",
        seq_len: int = 30,
        out_len: int = 3,
        enc_hidden: int = 64,
        dec_hidden: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | None = None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.out_len = out_len
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden or enc_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: nn.Module | None = None

    # -----------------------------------------------------
    # Data preparation: sliding window, multi-step targets
    # -----------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        """
        Creates input sequences of length seq_len and targets of length out_len.

        X: (B, seq_len, num_features)
        y: (B, out_len) – multi-step future returns.
        """
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

        num_features = X.shape[2]
        if self.model is None:
            self.model = MTSMFFBackbone(
                in_features=num_features,
                enc_hidden=self.enc_hidden,
                dec_hidden=self.dec_hidden,
                num_layers=self.num_layers,
                out_len=self.out_len,
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
            avg_loss = total_loss / len(loader)
            print(f"[{self.name}] Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")

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
        preds = self.predict(X_test)            # (B, out_len)
        # Flatten across horizon for simple metrics
        preds_flat = preds.reshape(-1)
        y_flat = y_test.reshape(-1)

        mse = float(np.mean((preds_flat - y_flat) ** 2))
        mae = float(np.mean(np.abs(preds_flat - y_flat)))
        rmse = float(mse ** 0.5)
        return {"mse": mse, "mae": mae, "rmse": rmse}

