from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Abstract blueprint for all models (trees, RNNs, CNNs, Transformers)."""

    def __init__(self, name: str):
        self.name = name 
        self.feature_cols = [
        "Return_t_1",
        "Return_t_2",
        "Return_t_3",
        "SMA_5",
        "SMA_10",
        "STD_5",
        ]

        self.target_col = "Return_t"

    
    # ---------- Data handling ----------
    @abstractmethod
    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a raw DataFrame (from your CSV) into (X, y) arrays suitable
        for this model. For sequence models this will build sliding windows.
        """
        pass

    # ---------- Core ML interface ----------
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for given features."""
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data and return a dict of metrics,
        e.g. {"mse": ..., "mae": ...}.
        """
        pass

