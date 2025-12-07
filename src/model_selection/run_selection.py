# src/model_selection/run_selection.py

"""
Entry point to run hyperparameter search for all models.

Usage (from inner project root, with PYTHONPATH=src):
    python -m model_selection.run_selection
"""

from pathlib import Path
import pandas as pd
import os
import sys

print("[run_selection] Module imported")  # <--- new: runs as soon as module loads

from .parameter_search import run_full_model_selection

# SRC_DIR = .../plfd-european-stock-predictor/plfd-european-stock-predictor/src
SRC_DIR = Path(__file__).resolve().parents[1]

# Features written by data_processing.py to:
#   src/data_processing/data/clean_features/STOXX600_features.csv
DATA_PATH = SRC_DIR / "data_processing" / "data" / "clean_features" / "STOXX600_features.csv"

# Results output relative to inner project root (parent of src)
PROJECT_ROOT = SRC_DIR.parent
OUTPUT_PATH = PROJECT_ROOT / "results" / "model_selection_results.json"


def main():
    print("[run_selection] Entered main()")  # ensure this appears
    print(f"[run_selection] CWD: {Path.cwd()}")
    print(f"[run_selection] DATA_PATH exists: {DATA_PATH.exists()} -> {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"[run_selection] CSV not found: {DATA_PATH}")

    print(f"[run_selection] Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    print(f"[run_selection] Loaded df shape: {df.shape}")

    # sort by date just in case
    df = df.sort_values("Date").reset_index(drop=True)

    print("[run_selection] Calling run_full_model_selection...")
    results = run_full_model_selection(
        df=df,
        k_folds=5,
        output_path=OUTPUT_PATH,
    )

    print("\n=== Best params per model ===")
    for name, info in results.items():
        print(f"{name}: RMSE={info['cv_rmse']:.6f}, params={info['best_params']}")


if __name__ == "__main__":
    main()
