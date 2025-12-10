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
    # Build X, y for the full dataset once
    tmp_model = model_class(**params)
    # --- NEW: use prepare_xy if the model provides it, else prepare_data ---
    if hasattr(tmp_model, "prepare_xy"):
        X, y = tmp_model.prepare_xy(df)
    else:
        X, y = tmp_model.prepare_data(df)
    # --- END NEW ---

    X = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X)

    if n_samples < 2:
        raise ValueError(
            f"Not enough samples for cross-validation (n_samples={n_samples})."
        )

    # Do not request more folds than samples
    effective_k = min(k, n_samples)
    if effective_k < 2:
        # We already ensured n_samples >= 2, so this is just a safety net
        effective_k = 2

    kf = KFold(n_splits=effective_k, shuffle=shuffle, random_state=random_state)

    rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        # Slice precomputed arrays; do NOT call prepare_data/prepare_xy again
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(**copy.deepcopy(params))

        # For models with prepare_xy/prepare_data, X_train/Y_train are already in
        # the correct array form expected by fit()/evaluate(), so just use them.
        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)

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
