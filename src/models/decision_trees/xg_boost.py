from __future__ import annotations
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.models.base_model import BaseModel
import numpy as np
import pandas as pd
from typing import Dict, Tuple


class XGBoostModel(BaseModel):

    def __init__(
        self,
        name: str = "XGBoost",
        n_estimators: int = 600,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,   # L2 regularization
        reg_alpha: float = 0.0,    # L1 regularization
        random_state: int = 42,
    ):
        super().__init__(name)

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=-1,
        )


    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert your engineered DataFrame into arrays suitable for training.
        """
        df_clean = df.dropna().copy()

        X = df_clean[self.feature_cols].astype(float).values
        y = df_clean[self.target_col].astype(float).values

        return X, y
    

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {
            "mse": mean_squared_error(y_test, preds),
            "mae": mean_absolute_error(y_test, preds),
            }


