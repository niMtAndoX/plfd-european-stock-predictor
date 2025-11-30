# random_forest_model.py

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from base_model import BaseModel   # <-- your abstract class goes here


class RandomForestModel(BaseModel):
    """
    Random Forest regression model that predicts the next-day return of
    a stock index using engineered features.
    """

def __init__(
    self,
    name: str = "RandomForest",
    n_estimators: int = 500,      # number of trees (VERY important)
    max_depth: int | None = 8,    # tree depth (controls overfitting)
    max_features: str | int | float | None = 3,  # features per split
    min_samples_split: int = 6,   # minimum samples to split a node
    min_samples_leaf: int = 3,    # leaves must have at least this many samples
    bootstrap: bool = True,       # bootstrap sampling (usually True)
    random_state: int = 42,       # reproducibility
):
    super().__init__(name)

    self.model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        random_state=random_state,
        n_jobs=-1  # use all CPU cores
    )


    # -------------------------------------------------------
    # Data preparation
    # -------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert your engineered DataFrame into arrays suitable for training.
        """
        df_clean = df.dropna().copy()

        X = df_clean[super.feature_cols].astype(float).values
        y = df_clean[super.target_col].astype(float).values

        return X, y

    # -------------------------------------------------------
    # Training
    # -------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """
        Fit the RandomForest model. X_val/y_val is unused but kept for
        compatibility with RNN/CNN/Transformer models.
        """
        self.model.fit(X_train, y_train)

    # -------------------------------------------------------
    # Prediction
    # -------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    # -------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X_test)

        return {
            "mse": mean_squared_error(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
        }

    