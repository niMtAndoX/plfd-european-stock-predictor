from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from base_model import BaseModel


# ---------------------------------------------------------
# Backbone: CNN -> LSTM -> FC
# ---------------------------------------------------------

class SACLSTMBackbone(nn.Module):
    def __init__(
        self,
        in_features: int,
        conv_channels: int = 32,
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = kernel_size // 2

        # 1D-CNN über die Zeitachse (Features als Channels)
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, conv_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM über die Zeit (batch_first=True: (B, T, C))
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # FC-Head: letzte Hidden-State -> Skalar (Return_t+1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        """
        # CNN erwartet (B, C_in, L) = (B, Features, Time)
        x = x.permute(0, 2, 1)          # (B, F, T)
        x = self.conv(x)                # (B, conv_channels, T)

        # zurück zu (B, T, C) für LSTM
        x = x.permute(0, 2, 1)          # (B, T, C)

        out, (h_n, c_n) = self.lstm(x)  # out: (B, T, H)
        last = out[:, -1, :]            # letzter Zeitschritt

        y = self.fc(last)               # (B, 1)
        return y


# ---------------------------------------------------------
# SACLSTMModel, das BaseModel implementiert
# ---------------------------------------------------------

class SACLSTMModel(BaseModel):

    def __init__(
        self,
        name: str = "SACLSTM",
        seq_len: int = 30,
        conv_channels: int = 32,
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | None = None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Wird in prepare_data gebaut, wenn wir die Feature-Anzahl kennen
        self.model: nn.Module | None = None

    # -----------------------------------------------------
    # Data Prep: Sliding Windows (wie bei CNN/SCINet)
    # -----------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        """
        Erzeugt Sequenzfenster aus deinem Feature-DataFrame.

        Input-DF: Spalten enthalten self.feature_cols + self.target_col.
        Output:
            X: (B, seq_len, num_features)
            y: (B,)
        """
        df_clean = df.dropna().copy()

        X_raw = df_clean[self.feature_cols].values.astype(float)
        y_raw = df_clean[self.target_col].values.astype(float)

        n = len(df_clean)
        num_samples = n - self.seq_len
        if num_samples <= 0:
            raise ValueError(
                f"Zu wenig Daten für seq_len={self.seq_len} (n={n})."
            )

        X_list = []
        y_list = []

        for i in range(num_samples):
            X_list.append(X_raw[i: i + self.seq_len])
            y_list.append(y_raw[i + self.seq_len])  # Vorhersage: nächster Tag

        X = np.array(X_list)  # (B, T, F)
        y = np.array(y_list)  # (B,)

        num_features = X.shape[2]

        if self.model is None:
            self.model = SACLSTMBackbone(
                in_features=num_features,
                conv_channels=self.conv_channels,
                kernel_size=self.kernel_size,
                lstm_hidden=self.lstm_hidden,
                lstm_layers=self.lstm_layers,
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

    