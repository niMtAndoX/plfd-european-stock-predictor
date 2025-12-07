
"""
Generic K-fold cross validation for any BaseModel subclass.

We assume:
- model_class(**params) constructs a model
- model.prepare_data(df) -> (X, y)
- model.fit(X_train, y_train)
- model.evaluate(X_val, y_val) -> dict with at least 'rmse'
"""

from __future__ import annotations
from typing import Dict, Any

import numpy as np
from sklearn.model_selection import KFold
import copy


def cross_validate_on_dataframe(
    model_class,
    params: Dict[str, Any],
    df,
    k: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> float:
    """
    Runs K-fold CV on a single dataframe for a given model + hyperparameters.

    Returns:
        mean_rmse over folds (we minimize this)
    """
    # Prepare full dataset once for these params (seq_len & out_len are inside params)
    # We use a temporary model instance just for data prep.
    tmp_model = model_class(**params)
    X, y = tmp_model.prepare_data(df)

    X = np.array(X)
    y = np.array(y)

    kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)

    rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # New model per fold to avoid weight leakage
        model = model_class(**copy.deepcopy(params))
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_val, y_val)

        rmse = metrics.get("rmse")
        if rmse is None:
            raise ValueError("Model.evaluate must return a dict with key 'rmse'.")

        rmses.append(rmse)
        print(f"    Fold {fold}: RMSE = {rmse:.6f}")

    mean_rmse = float(np.mean(rmses))
    print(f"    Mean CV RMSE: {mean_rmse:.6f}")
    return mean_rmse
