
"""
Train each model with its best hyperparameters on the training set,
then evaluate on a held-out test set.
"""

from __future__ import annotations
from typing import Dict, Any

from pathlib import Path
import json
import numpy as np
import pandas as pd

from model_selection.model_registry import MODEL_REGISTRY


def train_test_split_dataframe(df: pd.DataFrame, test_size: float = 0.2):
    """
    Simple chronological train/test split.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_size)
    df_train = df.iloc[:-n_test].reset_index(drop=True)
    df_test = df.iloc[-n_test:].reset_index(drop=True)
    return df_train, df_test


def evaluate_best_models(
    df: pd.DataFrame,
    selection_results_path: str | Path,
) -> Dict[str, Dict[str, float]]:
    """
    selection_results_path: JSON produced by run_full_model_selection
    """
    selection_results_path = Path(selection_results_path)
    with selection_results_path.open("r", encoding="utf-8") as f:
        selection_results = json.load(f)

    df_train, df_test = train_test_split_dataframe(df)

    leaderboard: Dict[str, Dict[str, float]] = {}

    for model_name, info in selection_results.items():
        best_params = info["best_params"]
        model_class = MODEL_REGISTRY[model_name]["class"]

        print(f"\n=== Training {model_name} with best params ===")
        model = model_class(**best_params)

        # prepare data using train df
        X_train, y_train = model.prepare_data(df_train)
        X_test, y_test = model.prepare_data(df_test)

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        leaderboard[model_name] = metrics
        print(f"{model_name} test metrics: {metrics}")

    return leaderboard


def save_leaderboard(leaderboard: Dict[str, Dict[str, float]], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # flatten to a table
    rows = []
    for model_name, metrics in leaderboard.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df_res = pd.DataFrame(rows)
    df_res.to_csv(path, index=False)
    print(f"Saved leaderboard to {path}")
