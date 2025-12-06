from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from base_model import BaseModel


# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------

def split_even_odd(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splitte entlang der Zeitachse in gerade / ungerade Indizes.
    x: (B, C, T)
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return x_even, x_odd


def merge_even_odd(x_even: torch.Tensor, x_odd: torch.Tensor, T: int) -> torch.Tensor:
    """
    Füge gerade/ungerade Sequenz wieder zu Länge T zusammen.
    Erwartet x_even/x_odd mit gleicher Batch/Channel-Form.
    """
    B, C, Te = x_even.shape
    _, _, To = x_odd.shape
    out = torch.zeros(B, C, T, device=x_even.device, dtype=x_even.dtype)

    out[..., 0::2] = x_even
    out[..., 1::2] = x_odd

    # Falls die Sequenz ungerade Länge hatte, wird der letzte Step von even/odd aufgefüllt.
    return out


# ---------------------------------------------------------
# SCI-Block (Split-Compress-Interpolate)
# ---------------------------------------------------------

class SCIBlock(nn.Module):
    """
    Vereinfachter SCI-Block wie im SCINet-Paper:
    - split in even/odd
    - wechselseitige Faltung + nichtlineare Kopplung
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()

        padding = kernel_size // 2

        # psi, phi (Multiplizierende Pfade)
        self.psi = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.Tanh(),
        )
        self.phi = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.Tanh(),
        )

        # eta, rho (Additive Pfade)
        self.eta = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.Tanh(),
        )
        self.rho = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.Tanh(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        x_even, x_odd = split_even_odd(x)

        # Interaktive Lernschritte (stark vereinfachte Version des Papers):
        psi_xe = self.psi(x_even)
        phi_xo = self.phi(x_odd)
        eta_xe = self.eta(x_even)
        rho_xo = self.rho(x_odd)

        h_odd = x_odd * torch.exp(psi_xe) + eta_xe
        h_even = x_even * torch.exp(phi_xo) + rho_xo

        h_odd = self.dropout(h_odd)
        h_even = self.dropout(h_even)

        # Rekombiniere zu ursprünglicher Länge
        T = x.size(-1)
        out = merge_even_odd(h_even, h_odd, T)
        return out


# ---------------------------------------------------------
# SCINet-Block: rekursive Anwendung über mehrere Levels
# ---------------------------------------------------------

class SCINetBlock(nn.Module):
    """
    Rekursiver SCINet-Baum über mehrere Levels.
    """

    def __init__(self, channels: int, levels: int = 2, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.levels = levels

        self.block = SCIBlock(channels, kernel_size=kernel_size, dropout=dropout)

        if levels > 1:
            self.sub_left = SCINetBlock(channels, levels=levels - 1,
                                        kernel_size=kernel_size, dropout=dropout)
            self.sub_right = SCINetBlock(channels, levels=levels - 1,
                                         kernel_size=kernel_size, dropout=dropout)
        else:
            self.sub_left = None
            self.sub_right = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        B, C, T = x.shape

        # Ein SCI-Block auf der aktuellen Ebene
        x_even, x_odd = split_even_odd(x)

        x_even = self.block(x_even)
        x_odd = self.block(x_odd)

        # Falls weitere Levels vorhanden sind: rekursiv
        if self.levels > 1:
            x_even = self.sub_left(x_even)
            x_odd = self.sub_right(x_odd)

        # Wieder zusammenführen auf Länge T
        out = merge_even_odd(x_even, x_odd, T)
        return out


# ---------------------------------------------------------
# Vollständiges SCINet-Modell (mehrere Stacks + FC-Head)
# ---------------------------------------------------------

class SCINetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        seq_len: int,
        stacks: int = 2,
        levels: int = 2,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.stacks = stacks

        # Input-Projektion auf hidden_channels
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # Mehrere SCINet-Blöcke (Stacks) mit Residual/Concat
        self.blocks = nn.ModuleList([
            SCINetBlock(hidden_channels, levels=levels,
                        kernel_size=kernel_size, dropout=dropout)
            for _ in range(stacks)
        ])

        self.norm = nn.LayerNorm([hidden_channels, seq_len])

        # FC-Head: flach machen und Skalar prognostizieren (Return_t+1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T)
        """
        x = self.input_proj(x)  # (B, hidden_channels, T)

        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual   # Residual-Verknüpfung

        x = self.norm(x)
        out = self.fc(x)      # (B, 1)
        return out


# ---------------------------------------------------------
# SCINetModel, das in dein BaseModel-Framework passt
# ---------------------------------------------------------

class SCINetModel(BaseModel):

    def __init__(
        self,
        name: str = "SCINet",
        seq_len: int = 30,
        stacks: int = 2,
        levels: int = 2,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | None = None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.stacks = stacks
        self.levels = levels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Wird dynamisch gebaut, wenn wir num_features kennen
        self.model: nn.Module | None = None

    # -----------------------------------------------------
    # Data Prep: Sliding Windows + (B, C, T)-Format
    # -----------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        df_clean = df.dropna().copy()

        X_raw = df_clean[self.feature_cols].values.astype(float)
        y_raw = df_clean[self.target_col].values.astype(float)

        n = len(df_clean)
        num_samples = n - self.seq_len
        if num_samples <= 0:
            raise ValueError(f"Zu wenig Daten für seq_len={self.seq_len} (n={n})")

        X_list = []
        y_list = []
        for i in range(num_samples):
            X_list.append(X_raw[i : i + self.seq_len])
            y_list.append(y_raw[i + self.seq_len])  # nächster Tag

        X = np.array(X_list)                        # (B, T, F)
        y = np.array(y_list)                        # (B,)

        # In (B, C, T) transponieren: C = Features
        X = np.transpose(X, (0, 2, 1))

        num_features = X.shape[1]
        if self.model is None:
            self.model = SCINetBackbone(
                in_channels=num_features,
                seq_len=self.seq_len,
                stacks=self.stacks,
                levels=self.levels,
                hidden_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
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
    # Vorhersage
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

    