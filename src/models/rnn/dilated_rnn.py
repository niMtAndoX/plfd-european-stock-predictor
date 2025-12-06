from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from base_model import BaseModel


# ---------------------------------------------------------
# Dilated GRU layer
# ---------------------------------------------------------

class DilatedGRULayer(nn.Module):
    """
    A single dilated GRU layer.
    Applies GRUCell over time, but steps are spaced by dilation.
    """

    def __init__(self, input_size, hidden_size, dilation=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dilation = dilation
        self.gru_cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, x):
        """
        x: (B, T, F)
        returns:
            output: (B, T, H)
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)

        outputs = torch.zeros(B, T, self.hidden_size, device=x.device)

        # Iterate through time steps with dilation
        for t in range(T):
            if t % self.dilation == 0:
                h = self.gru_cell(x[:, t, :], h)
            outputs[:, t, :] = h

        return outputs


# ---------------------------------------------------------
# Dilated RNN backbone (stack of dilated layers)
# ---------------------------------------------------------

class DilatedRNNBackbone(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dilations=None,
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 4]

        layers = []
        in_size = input_size
        for d in dilations:
            layers.append(DilatedGRULayer(in_size, hidden_size, dilation=d))
            in_size = hidden_size

        self.layers = nn.ModuleList(layers)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        x: (B, T, F)
        """

        out = x
        for layer in self.layers:
            out = layer(out)

        # Last time-step output is used for forecasting
        last = out[:, -1, :]  # (B, H)
        return self.fc(last)


# ---------------------------------------------------------
# Model wrapper compatible with BaseModel
# ---------------------------------------------------------

class DilatedRNNModel(BaseModel):
    def __init__(
        self,
        name: str = "DilatedRNN",
        seq_len: int = 30,
        hidden_size: int = 64,
        dilations=None,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | None = None,
    ):
        super().__init__(name)

        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dilations = dilations or [1, 2, 4]
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None

    # -----------------------------------------------------
    # Sliding window data preparation
    # -----------------------------------------------------
    def prepare_data(self, df: pd.DataFrame):
        df = df.dropna().copy()

        X_raw = df[self.feature_cols].values.astype(float)
        y_raw = df[self.target_col].values.astype(float)

        n = len(df)
        num_samples = n - self.seq_len

        X_list, y_list = [], []
        for i in range(num_samples):
            X_list.append(X_raw[i : i + self.seq_len])
            y_list.append(y_raw[i + self.seq_len])

        X = np.array(X_list)  # (B, T, F)
        y = np.array(y_list)

        num_features = X.shape[2]

        if self.model is None:
            self.model = DilatedRNNBackbone(
                input_size=num_features,
                hidden_size=self.hidden_size,
                dilations=self.dilations,
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
