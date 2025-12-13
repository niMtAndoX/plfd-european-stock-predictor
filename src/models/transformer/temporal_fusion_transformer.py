from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.base_model import BaseModel


# ---------------------------------------------------------
# Positional Encoding (sine/cosine, like standard Transformer)
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
# Gated Residual Network (GRN) as in TFT
# ---------------------------------------------------------

class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        context_size: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.context_size = context_size

        self.lin_in = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.lin_ctx = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.lin_ctx = None

        self.elu = nn.ELU()
        self.lin_out = nn.Linear(hidden_size, self.output_size)

        # Gating (GLU)
        self.gate = nn.Linear(self.output_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

        # Residual projection if needed
        if self.output_size != input_size:
            self.skip_proj = nn.Linear(input_size, self.output_size)
        else:
            self.skip_proj = None

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None):
        """
        x: (B, T, D) or (B, D)
        context: optional, same batch/seq shape, last dim = context_size
        """
        original_x = x

        h = self.lin_in(x)
        if self.lin_ctx is not None and context is not None:
            h = h + self.lin_ctx(context)
        h = self.elu(h)
        h = self.lin_out(h)

        # Gating
        gate = self.sigmoid(self.gate(h))
        h = self.dropout(h * gate)

        # Residual
        if self.skip_proj is not None:
            residual = self.skip_proj(original_x)
        else:
            residual = original_x

        return self.norm(h + residual)


# ---------------------------------------------------------
# TFT Backbone: LSTM encoder + temporal attention + GRNs
# ---------------------------------------------------------

class TFTBackbone(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        nhead: int = 4,
        attn_dropout: float = 0.1,
        grn_dropout: float = 0.1,
        out_len: int = 1,
    ):
        super().__init__()

        self.d_model = d_model
        self.out_len = out_len

        # Variable selection could be more complex; here: simple linear embed
        self.input_proj = nn.Linear(in_features, d_model)

        self.position = PositionalEncoding(d_model, dropout=grn_dropout)

        # LSTM encoder (temporal processing)
        self.encoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Project LSTM outputs to attention dimension
        self.enc_proj = nn.Linear(lstm_hidden, d_model)

        # Multi-head temporal attention (Masked Interpretable Multi-Head Attention)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
            batch_first=True,  # (B, T, E)
        )

        # GRNs for post-attention & output fusion
        self.grn_attn = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            output_size=d_model,
            dropout=grn_dropout,
        )
        self.grn_out = GatedResidualNetwork(
            input_size=d_model,
            hidden_size=d_model,
            output_size=d_model,
            dropout=grn_dropout,
        )

        # Final prediction head (per future step)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T_in, F)
        returns:
            y_hat: (B, out_len)
        For out_len > 1, we just reuse the same temporal attention context
        and decode multiple steps with a learned projection.
        """
        B, T, F = x.shape

        # 1) Variable selection / embedding + positional encoding
        h = self.input_proj(x)         # (B, T, d_model)
        h = self.position(h)           # (B, T, d_model)

        # 2) LSTM encoder
        enc_out, (h_n, c_n) = self.encoder_lstm(h)  # enc_out: (B, T, H_lstm)
        enc_out = self.enc_proj(enc_out)            # (B, T, d_model)

        # 3) Temporal attention (query: final timestep)
        query = enc_out[:, -1:, :]                  # (B, 1, d_model)
        attn_out, _ = self.attn(query, enc_out, enc_out)  # (B, 1, d_model)

        # 4) GRN over attention output (fusion)
        attn_out = self.grn_attn(attn_out)          # (B, 1, d_model)

        # 5) Decode out_len steps (simple repetition + GRN + FC)
        #    In full TFT, decoder is more complex; here we keep it simple.
        rep = attn_out.repeat(1, self.out_len, 1)   # (B, out_len, d_model)
        rep = self.grn_out(rep)                     # (B, out_len, d_model)

        y_hat = self.fc_out(rep).squeeze(-1)        # (B, out_len)
        return y_hat
        

# ---------------------------------------------------------
# TFTModel wrapper (BaseModel)
# ---------------------------------------------------------

class TFTModel(BaseModel):
    def __init__(
        self,
        name: str = "TFT",
        seq_len: int = 30,
        out_len: int = 1,
        d_model: int = 64,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        nhead: int = 4,
        attn_dropout: float = 0.1,
        grn_dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | None = None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.out_len = out_len
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.nhead = nhead
        self.attn_dropout = attn_dropout
        self.grn_dropout = grn_dropout
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

        X = np.array(X_list)  # (B, T_in, F)
        y = np.array(y_list)  # (B, out_len)

        num_features = X.shape[2]
        if self.model is None:
            self.model = TFTBackbone(
                in_features=num_features,
                d_model=self.d_model,
                lstm_hidden=self.lstm_hidden,
                lstm_layers=self.lstm_layers,
                nhead=self.nhead,
                attn_dropout=self.attn_dropout,
                grn_dropout=self.grn_dropout,
                out_len=self.out_len,
            ).to(self.device)

        return X, y

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Ensure model is initialized even when prepare_data() was not called on this instance
        if self.model is None:
            if X_train.ndim != 3:
                raise ValueError(
                    f"{self.name}.fit expected X_train with shape (B, T, F), "
                    f"got {X_train.shape}"
                )
            num_features = X_train.shape[2]
            self.model = TFTBackbone(
                in_features=num_features,
                d_model=self.d_model,
                lstm_hidden=self.lstm_hidden,
                lstm_layers=self.lstm_layers,
                nhead=self.nhead,
                attn_dropout=self.attn_dropout,
                grn_dropout=self.grn_dropout,
                out_len=self.out_len,
            ).to(self.device)

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
                preds = self.model(Xb)          # (B, out_len)
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
        preds = self.predict(X_test)
        preds_flat = preds.reshape(-1)
        y_flat = y_test.reshape(-1)

        mse = float(np.mean((preds_flat - y_flat) ** 2))
        mae = float(np.mean(np.abs(preds_flat - y_flat)))
        rmse = float(mse ** 0.5)
        return {"mse": mse, "mae": mae, "rmse": rmse}