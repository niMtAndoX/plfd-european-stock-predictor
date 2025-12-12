"""
Train each model with its best hyperparameters on the training set,
then evaluate on a held-out test set, using the combined
STOXX600_with_asian_features.csv produced by the ETL step.
"""

from __future__ import annotations
from typing import Dict

from pathlib import Path
import json
import pandas as pd

from src.model_selection.model_registry import MODEL_REGISTRY


def train_test_split_dataframe(df: pd.DataFrame, test_size: float = 0.2):
    """Simple chronological train/test split by Date."""
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_size)
    df_train = df.iloc[:-n_test].reset_index(drop=True)
    df_test = df.iloc[-n_test:].reset_index(drop=True)
    return df_train, df_test


def load_stoxx600_with_asian_features(clean_dir: Path) -> pd.DataFrame:
    """
    Load the combined STOXX600 + aggregated Asian features dataset.
    """
    csv_path = clean_dir / "STOXX600_with_asian_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"[evaluate_models] Combined features CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def evaluate_best_models(
    df: pd.DataFrame,
    selection_results_path: str | Path,
) -> Dict[str, Dict[str, float]]:
    """
    selection_results_path: JSON produced by run_full_model_selection.
    """
    selection_results_path = Path(selection_results_path)
    with selection_results_path.open("r", encoding="utf-8") as f:
        selection_results = json.load(f)

    df_train, df_test = train_test_split_dataframe(df)

    leaderboard: Dict[str, Dict[str, float]] = {}

    for model_name, info in selection_results.items():
        if model_name not in MODEL_REGISTRY:
            # skip any non-model keys
            continue

        best_params = info["best_params"]
        model_class = MODEL_REGISTRY[model_name]["class"]

        print(f"\n[evaluate_models] Training {model_name} with best params")
        model = model_class(**best_params)

        X_train, y_train = model.prepare_data(df_train)
        X_test, y_test = model.prepare_data(df_test)

        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        leaderboard[model_name] = metrics
        print(f"[evaluate_models] {model_name} test metrics: {metrics}")

    return leaderboard


def save_leaderboard(leaderboard: Dict[str, Dict[str, float]], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, metrics in leaderboard.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df_res = pd.DataFrame(rows)
    df_res.to_csv(path, index=False)
    print(f"[evaluate_models] Saved leaderboard to {path}")


def main():
    """
    Entry point called by run_full_pipeline.py and the Colab notebook.
    """
    src_dir = Path(__file__).resolve().parents[1]   # .../src
    project_root = src_dir.parent                  # inner project root

    clean_dir = src_dir / "data_processing" / "data" / "clean_features"
    selection_json = project_root / "results" / "model_selection_results.json"
    leaderboard_csv = project_root / "results" / "leaderboard.csv"

    print(f"[evaluate_models] clean_dir:      {clean_dir}")
    print(f"[evaluate_models] selection_json: {selection_json}")
    print(f"[evaluate_models] leaderboard_csv:{leaderboard_csv}")

    if not selection_json.exists():
        raise FileNotFoundError(f"[evaluate_models] Selection JSON not found: {selection_json}")

    df = load_stoxx600_with_asian_features(clean_dir)

    leaderboard = evaluate_best_models(df, selection_json)
    save_leaderboard(leaderboard, leaderboard_csv)


if __name__ == "__main__":
    main()
