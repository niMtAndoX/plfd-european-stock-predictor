"""
Entry point to run hyperparameter search for all models.

Usage (from inner project root, with PYTHONPATH=src):
    python -m model_selection.run_selection
"""

from pathlib import Path
import pandas as pd
import os
import sys
import json  # <-- needed for JSON load/save

print("[run_selection] Module imported")  

from .parameter_search import run_full_model_selection

# SRC_DIR = .../plfd-european-stock-predictor/plfd-european-stock-predictor/src
SRC_DIR = Path(__file__).resolve().parents[1]

# Features written by data_processing.py to:
#   src/data_processing/data/clean_features/merged_features.csv
# This contains Asian market features (KOSPI, SSE, TAIEX, NIKKEI225) for predicting STOXX600
DATA_PATH = SRC_DIR / "data_processing" / "data" / "clean_features" / "merged_features.csv"

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

    # ------------------------------------------------------------------
    # Determine the globally best model from this run
    # ------------------------------------------------------------------
    best_model_name = min(
        results.keys(),
        key=lambda m: results[m]["cv_rmse"],
    )
    best_entry = results[best_model_name]

    run_best_record = {
        "model_name": best_model_name,
        "cv_rmse": best_entry["cv_rmse"],
        "best_params": best_entry["best_params"],
    }

    # ------------------------------------------------------------------
    # Append / merge into JSON file without losing previous content
    # ------------------------------------------------------------------
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            stored = json.load(f)
    else:
        stored = {}

    # Merge current per-model results into stored dict
    # (keeps previous models unless overwritten by same key)
    stored.update(results)

    # Maintain a history of global bests
    history = stored.get("global_best_history", [])
    history.append(run_best_record)
    stored["global_best_history"] = history

    # Also store the currently best model across all runs for convenience
    # (lowest cv_rmse in history)
    overall_best = min(history, key=lambda r: r["cv_rmse"])
    stored["global_best"] = overall_best

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(stored, f, indent=2)

    print(
        f"[run_selection] Global best this run: {best_model_name} "
        f"(cv_rmse={best_entry['cv_rmse']:.6f})"
    )


if __name__ == "__main__":
    main()
